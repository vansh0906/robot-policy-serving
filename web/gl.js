// Small WebGL renderer for the arm viewer: lit cylinders and spheres built on the CPU each frame, a ground grid,
// and an orbit camera. World frame is z-up, metres. No dependencies.
"use strict";

const VS = `
attribute vec3 aPos; attribute vec3 aNrm;
uniform mat4 uVP; varying vec3 vN; varying vec3 vP;
void main(){ vN = aNrm; vP = aPos; gl_Position = uVP * vec4(aPos, 1.0); }`;
const FS = `
precision mediump float;
uniform vec4 uColor; uniform float uLit; uniform vec3 uEye; varying vec3 vN; varying vec3 vP;
void main(){
  vec3 n = normalize(vN); vec3 l = normalize(vec3(0.4, -0.5, 0.9));
  float d = max(dot(n, l), 0.0); vec3 v = normalize(uEye - vP); vec3 h = normalize(l + v);
  float s = pow(max(dot(n, h), 0.0), 32.0) * 0.35;
  vec3 c = mix(uColor.rgb, uColor.rgb * (0.35 + 0.65 * d) + s, uLit);
  gl_FragColor = vec4(c, uColor.a);
}`;

function mat4Mul(a, b) {            // column-major 4x4
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
  }
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(e, t, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(e, t)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]);
}
function invert4(m) {               // general 4x4 inverse (for picking)
  const inv = new Float32Array(16);
  inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
  inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
  inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
  inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
  inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
  inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
  inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
  inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
  inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
  inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
  inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
  inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
  inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  det = det ? 1 / det : 0;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

class Mesh {                       // accumulates triangles (position + normal) for one colour
  constructor() { this.p = []; this.n = []; }
  tri(a, b, c, na, nb, nc) { this.p.push(...a, ...b, ...c); this.n.push(...na, ...nb, ...nc); }
}

function cylinder(mesh, a, b, r, seg = 18) {
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L = Math.hypot(...d) || 1e-9;
  const w = [d[0] / L, d[1] / L, d[2] / L];
  const tmp = Math.abs(w[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let u = [w[1] * tmp[2] - w[2] * tmp[1], w[2] * tmp[0] - w[0] * tmp[2], w[0] * tmp[1] - w[1] * tmp[0]];
  const ul = Math.hypot(...u); u = u.map(x => x / ul);
  const v = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]];
  const ring = (c, t) => { const ct = Math.cos(t), st = Math.sin(t);
    const n = [u[0] * ct + v[0] * st, u[1] * ct + v[1] * st, u[2] * ct + v[2] * st];
    return [[c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2]], n]; };
  const nw = w.map(x => -x);
  for (let i = 0; i < seg; i++) {
    const t0 = 2 * Math.PI * i / seg, t1 = 2 * Math.PI * (i + 1) / seg;
    const [a0, n0] = ring(a, t0), [a1, n1] = ring(a, t1), [b0] = ring(b, t0), [b1] = ring(b, t1);
    mesh.tri(a0, a1, b1, n0, n1, n1); mesh.tri(a0, b1, b0, n0, n1, n0);
    mesh.tri(a, a1, a0, nw, nw, nw); mesh.tri(b, b0, b1, w, w, w);
  }
}

function sphere(mesh, c, r, lat = 12, lon = 18) {
  const pt = (i, j) => { const th = Math.PI * i / lat, ph = 2 * Math.PI * j / lon;
    const n = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    return [[c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2]], n]; };
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
    const [p00, n00] = pt(i, j), [p10, n10] = pt(i + 1, j), [p01, n01] = pt(i, j + 1), [p11, n11] = pt(i + 1, j + 1);
    mesh.tri(p00, p10, p11, n00, n10, n11); mesh.tri(p00, p11, p01, n00, n11, n01);
  }
}

class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) throw new Error("WebGL not available");
    this.gl = gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    this.prog = p;
    this.loc = { pos: gl.getAttribLocation(p, "aPos"), nrm: gl.getAttribLocation(p, "aNrm"),
      vp: gl.getUniformLocation(p, "uVP"), color: gl.getUniformLocation(p, "uColor"),
      lit: gl.getUniformLocation(p, "uLit"), eye: gl.getUniformLocation(p, "uEye") };
    this.pbuf = gl.createBuffer(); this.nbuf = gl.createBuffer();
    this.cam = { az: -0.9, el: 0.55, dist: 1.9, target: [0, 0, 0.3] };
    this.grid = this.makeGrid();
  }
  makeGrid() {
    const p = [], n = [];
    for (let i = -10; i <= 10; i++) {
      const s = i * 0.1;
      p.push(s, -1, 0, s, 1, 0, -1, s, 0, 1, s, 0);
      for (let k = 0; k < 4; k++) n.push(0, 0, 1);
    }
    return { p: new Float32Array(p), n: new Float32Array(n) };
  }
  eye() {
    const c = this.cam;
    return [c.target[0] + c.dist * Math.cos(c.el) * Math.cos(c.az), c.target[1] + c.dist * Math.cos(c.el) * Math.sin(c.az),
      c.target[2] + c.dist * Math.sin(c.el)];
  }
  viewProj() {
    const w = this.canvas.width, h = this.canvas.height;
    return mat4Mul(perspective(0.8, w / Math.max(h, 1), 0.05, 20), lookAt(this.eye(), this.cam.target, [0, 0, 1]));
  }
  // Ray from a canvas pixel, intersected with the horizontal plane z = h; returns [x, y, h] or null.
  pick(px, py, h) {
    const w = this.canvas.clientWidth, H = this.canvas.clientHeight;
    const nx = 2 * px / w - 1, ny = 1 - 2 * py / H;
    const inv = invert4(this.viewProj());
    const un = (z) => { const v = [nx, ny, z, 1], o = [0, 0, 0, 0];
      for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[r] += inv[k * 4 + r] * v[k];
      return [o[0] / o[3], o[1] / o[3], o[2] / o[3]]; };
    const a = un(-1), b = un(1), dz = b[2] - a[2];
    if (Math.abs(dz) < 1e-9) return null;
    const t = (h - a[2]) / dz;
    if (t < 0) return null;
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), h];
  }
  draw(items) {                     // items: [{mesh, color:[r,g,b,a]}]
    const gl = this.gl, dpr = window.devicePixelRatio || 1;
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const bg = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim() || "#0f1720";
    const c = bg.match(/\w\w/g).map(x => parseInt(x, 16) / 255);
    gl.clearColor(c[0], c[1], c[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.loc.vp, false, this.viewProj());
    gl.uniform3fv(this.loc.eye, this.eye());
    const put = (p, n) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.pbuf); gl.bufferData(gl.ARRAY_BUFFER, p, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.nbuf); gl.bufferData(gl.ARRAY_BUFFER, n, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.nrm); gl.vertexAttribPointer(this.loc.nrm, 3, gl.FLOAT, false, 0, 0);
    };
    put(this.grid.p, this.grid.n);
    gl.uniform1f(this.loc.lit, 0); gl.uniform4fv(this.loc.color, [0.45, 0.55, 0.65, 0.35]);
    gl.drawArrays(gl.LINES, 0, this.grid.p.length / 3);
    for (const it of items) {
      if (it.lines) {
        put(new Float32Array(it.lines), new Float32Array(it.lines.length).fill(0));
        gl.uniform1f(this.loc.lit, 0); gl.uniform4fv(this.loc.color, it.color);
        gl.drawArrays(gl.LINE_STRIP, 0, it.lines.length / 3);
        continue;
      }
      put(new Float32Array(it.mesh.p), new Float32Array(it.mesh.n));
      gl.uniform1f(this.loc.lit, 1); gl.uniform4fv(this.loc.color, it.color);
      gl.drawArrays(gl.TRIANGLES, 0, it.mesh.p.length / 3);
    }
  }
}

window.ArmGL = { Renderer, Mesh, cylinder, sphere };
