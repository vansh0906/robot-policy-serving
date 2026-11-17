// Viewer app: live robot state over /ws/viewer, robot control through the HTTP API, episode replay.
"use strict";
const $ = id => document.getElementById(id);
const R = new ArmGL.Renderer($("gl"));
let ARM = null, MODELS = [], robots = [], selected = null, trail = [], rttHist = {}, replay = null, ws = null;

// ---------- kinematics (same as rps/arm.py) ----------
function linkPoints(q) {
  const [q0, q1, q2] = q, c = Math.cos(q0), s = Math.sin(q0);
  const re = ARM.L1 * Math.cos(q1), ze = ARM.H0 + ARM.L1 * Math.sin(q1);
  const rt = re + ARM.L2 * Math.cos(q1 + q2), zt = ze + ARM.L2 * Math.sin(q1 + q2);
  return [[0, 0, 0], [0, 0, ARM.H0], [re * c, re * s, ze], [rt * c, rt * s, zt]];
}
function ikOk(p) {                 // elbow-up IK inside joint limits?
  const r = Math.hypot(p[0], p[1]), h = p[2] - ARM.H0;
  const c2 = (r * r + h * h - ARM.L1 ** 2 - ARM.L2 ** 2) / (2 * ARM.L1 * ARM.L2);
  if (Math.abs(c2) > 1) return false;
  const q2 = -Math.acos(c2), q1 = Math.atan2(h, r) - Math.atan2(ARM.L2 * Math.sin(q2), ARM.L1 + ARM.L2 * Math.cos(q2));
  const q = [Math.atan2(p[1], p[0]), q1, q2];
  return q.every((v, i) => v >= ARM.q_lo[i] && v <= ARM.q_hi[i]);
}

// ---------- server I/O ----------
async function api(method, path, body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const j = await r.json();
  if (!r.ok) throw new Error(j.msg || j.error || r.statusText);
  return j;
}
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/viewer`);
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "pill on"; };
  ws.onclose = () => { $("conn").textContent = "disconnected – retrying"; $("conn").className = "pill off"; setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello") { ARM = m.arm; MODELS = m.models; fillModels(); }
    else if (m.type === "state") onState(m.robots);
    else if (m.type === "error") $("selErr").textContent = m.msg;
  };
}
function fillModels() {
  for (const sel of [$("newModel"), $("selModel")]) {
    sel.innerHTML = "";
    for (const c of MODELS) {
      const o = document.createElement("option");
      o.value = c.name; o.textContent = `${c.name} (K=${c.K})${c.deployable ? "" : " – blocked"}`;
      sel.appendChild(o);
    }
  }
  const dep = MODELS.filter(c => c.deployable).map(c => c.name);
  if (dep.length) $("newModel").value = dep[dep.length - 1];
}

function onState(list) {
  robots = list;
  if (!selected && robots.length) select(robots[0].id);
  if (selected && !robots.find(r => r.id === selected)) selected = null;
  for (const r of robots) {
    const h = rttHist[r.id] || (rttHist[r.id] = []);
    if (r.rtt_p50 != null) { h.push(r.rtt_p50); if (h.length > 200) h.shift(); }
  }
  // update rows in place: rebuilding the table 20 times a second would swallow clicks
  const tb = $("robots").querySelector("tbody");
  const ids = robots.map(r => r.id).join(",");
  if (tb.dataset.ids !== ids) {
    tb.dataset.ids = ids;
    tb.innerHTML = robots.map(r => `<tr data-id="${r.id}"><td>${r.id}</td><td></td><td></td><td></td></tr>`).join("");
  }
  robots.forEach((r, i) => {
    const tr = tb.rows[i], cells = [r.model, r.rtt_p50 == null ? "–" : r.rtt_p50.toFixed(1), `${r.successes}/${r.successes + r.timeouts}`];
    cells.forEach((v, k) => { if (tr.cells[k + 1].textContent !== v) tr.cells[k + 1].textContent = v; });
    tr.classList.toggle("sel", r.id === selected);
  });
  const r = robots.find(x => x.id === selected);
  if (r) {
    if (!replay) {
      trail.push(r.tip); if (trail.length > 150) trail.shift();
      $("status").innerHTML = `<b>${r.id}</b> · ${r.model} · ${r.status}` +
        (r.last_error ? ` · <span class="bad">last reject: ${r.last_error}</span>` : "");
    }
    showSelected(r);
  }
}
function select(id) {
  selected = id; trail = [];
  const r = robots.find(x => x.id === id);
  $("selControls").hidden = !r; $("selTitle").textContent = r ? r.id : "No robot selected";
  if (r) {
    $("selModel").value = r.model; $("selDelay").value = r.delay_ms; $("selJitter").value = r.jitter_ms;
    $("selAuto").checked = r.auto;
    [$("tx").value, $("ty").value, $("tz").value] = r.target.map(v => v.toFixed(2));
  }
}
function showSelected(r) {
  $("selDelayV").textContent = `${r.delay_ms} ms`; $("selJitterV").textContent = `${r.jitter_ms} ms`;
  if (document.activeElement !== $("selModel")) $("selModel").value = r.model;
  $("selStats").innerHTML = `<dt>Status</dt><dd>${r.status}</dd><dt>Chunk</dt><dd>${r.K} step${r.K > 1 ? "s" : ""}</dd>` +
    `<dt>RTT p50 / p99</dt><dd>${fmt(r.rtt_p50)} / ${fmt(r.rtt_p99)} ms</dd>` +
    `<dt>Reached</dt><dd class="ok">${r.successes}</dd><dt>Timed out</dt><dd class="${r.timeouts ? "bad" : ""}">${r.timeouts}</dd>` +
    `<dt>Rejected</dt><dd>${r.rejects}</dd>`;
  drawRtt(rttHist[r.id] || []);
}
const fmt = v => v == null ? "–" : v.toFixed(1);
function drawRtt(h) {
  const c = $("rtt"), g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  c.width = c.clientWidth * dpr; c.height = 90 * dpr; g.scale(dpr, dpr);
  const W = c.clientWidth, H = 90, max = Math.max(25, ...h) * 1.15;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "#233142"; g.fillStyle = "#8a9bb0"; g.font = "10px system-ui";
  for (const v of [20, max / 2]) { const y = H - v / max * H; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
    g.fillText(v === 20 ? "20 ms (1 control period)" : `${v.toFixed(0)}`, 4, y - 2); }
  g.strokeStyle = "#4fb3ff"; g.lineWidth = 1.5; g.beginPath();
  h.forEach((v, i) => { const x = i / 199 * W, y = H - v / max * H; i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}

// ---------- rendering ----------
function frame() {
  const items = [];
  let q = null, target = null, tr = trail;
  if (replay) { q = replay.ep.frames[replay.i]; target = replay.ep.target; tr = replay.trail; }
  else { const r = robots.find(x => x.id === selected); if (r) { q = r.q; target = r.target; } }
  if (ARM && q) {
    const P = linkPoints(q);
    const links = new ArmGL.Mesh(), joints = new ArmGL.Mesh(), base = new ArmGL.Mesh(), tgt = new ArmGL.Mesh();
    ArmGL.cylinder(base, [0, 0, 0], [0, 0, 0.04], 0.09);
    ArmGL.cylinder(links, P[0], P[1], 0.035); ArmGL.cylinder(links, P[1], P[2], 0.028); ArmGL.cylinder(links, P[2], P[3], 0.022);
    for (const p of P.slice(1)) ArmGL.sphere(joints, p, 0.042);
    ArmGL.sphere(tgt, target, 0.025);
    items.push({ mesh: base, color: [0.25, 0.3, 0.36, 1] }, { mesh: links, color: [0.85, 0.88, 0.92, 1] },
      { mesh: joints, color: [0.31, 0.7, 1.0, 1] }, { mesh: tgt, color: [0.24, 0.81, 0.57, 0.9] });
    if (tr.length > 1) items.push({ lines: tr.flat(), color: [1.0, 0.75, 0.3, 0.9] });
  }
  R.draw(items);
  if (replay && replay.playing) {
    replay.acc += 1;
    if (replay.acc >= 2) {          // frames are 40 ms apart; ~60 fps display
      replay.acc = 0;
      if (replay.i < replay.ep.frames.length - 1) { replay.i++; updateReplay(); } else replay.playing = false;
    }
  }
  requestAnimationFrame(frame);
}

// ---------- camera & picking ----------
let drag = null;
$("gl").addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, moved: false }; });
window.addEventListener("mouseup", async e => {
  if (drag && !drag.moved && e.target === $("gl") && selected && !replay) {
    const rect = $("gl").getBoundingClientRect();
    const p = R.pick(e.clientX - rect.left, e.clientY - rect.top, parseFloat($("tz").value) || 0.3);
    if (p) { [$("tx").value, $("ty").value] = [p[0].toFixed(2), p[1].toFixed(2)]; sendTarget(); }
  }
  drag = null;
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (drag.moved) { R.cam.az -= dx * 0.008; R.cam.el = Math.max(0.05, Math.min(1.5, R.cam.el + dy * 0.008)); drag.x = e.clientX; drag.y = e.clientY; }
});
$("gl").addEventListener("wheel", e => { e.preventDefault(); R.cam.dist = Math.max(0.6, Math.min(5, R.cam.dist * (1 + e.deltaY * 0.001))); }, { passive: false });

// ---------- controls ----------
$("launch").onclick = async () => {
  $("launchErr").textContent = "";
  try {
    const r = await api("POST", "/api/robots", { model: $("newModel").value, delay_ms: +$("newDelay").value, jitter_ms: +$("newJitter").value });
    select(r.id);
  } catch (e) { $("launchErr").textContent = e.message; }
};
$("robots").addEventListener("click", e => { const tr = e.target.closest("tr[data-id]"); if (tr) { stopReplay(); select(tr.dataset.id); } });
async function config(body) {
  $("selErr").textContent = "";
  try { await api("POST", `/api/robots/${selected}/config`, body); } catch (e) { $("selErr").textContent = e.message; }
}
$("selModel").onchange = () => config({ model: $("selModel").value });
$("selDelay").onchange = () => config({ delay_ms: +$("selDelay").value });
$("selJitter").onchange = () => config({ jitter_ms: +$("selJitter").value });
$("selAuto").onchange = () => config({ auto: $("selAuto").checked });
function sendTarget() {
  const t = [+$("tx").value, +$("ty").value, +$("tz").value];
  $("selErr").textContent = ARM && !ikOk(t) ? "That target is outside the reachable workspace." : "";
  if (ws && ws.readyState === 1 && selected) ws.send(JSON.stringify({ type: "set_target", robot: selected, target: t }));
  $("selAuto").checked = false;
}
$("sendTarget").onclick = sendTarget;
$("randTarget").onclick = () => {
  for (let i = 0; i < 200; i++) {
    const r = 0.25 + Math.random() * 0.35, a = (Math.random() * 2 - 1) * 2.3, z = 0.05 + Math.random() * 0.5;
    const p = [r * Math.cos(a), r * Math.sin(a), z];
    if (ikOk(p)) { [$("tx").value, $("ty").value, $("tz").value] = p.map(v => v.toFixed(2)); return sendTarget(); }
  }
};

// ---------- episodes & replay ----------
async function refreshEpisodes() {
  try {
    const eps = await api("GET", "/api/episodes");
    const tb = $("episodes").querySelector("tbody"), key = eps.slice(0, 25).map(e => e.id).join(",");
    if (tb.dataset.key !== key) tb.dataset.key = key, tb.innerHTML = eps.slice(0, 25).map(e => `<tr data-ep="${e.id}"><td>${e.id}</td><td>${e.robot}</td>` +
      `<td>${e.model}</td><td>${e.delay_ms}</td><td class="${e.success ? "ok" : "bad"}">${e.success ? "reached " + e.duration_s + " s" : "missed"}</td></tr>`).join("");
    const m = await api("GET", "/api/metrics");
    $("metrics").textContent = `${m.counters.requests || 0} requests · batch mean ${m.batch_size ? m.batch_size.mean : "–"} · ` +
      `inference p50 ${m.infer_ms ? m.infer_ms.p50 : "–"} ms · server p99 ${m.server_ms ? m.server_ms.p99 : "–"} ms`;
  } catch (e) { /* server restarting */ }
}
$("episodes").addEventListener("click", async e => {
  const tr = e.target.closest("tr[data-ep]"); if (!tr) return;
  const ep = await api("GET", `/api/episodes/${tr.dataset.ep}`);
  if (!ep.frames.length) return;
  replay = { ep, i: 0, playing: true, acc: 0, trail: [] };
  $("replayBar").hidden = false; $("scrub").max = ep.frames.length - 1; updateReplay();
});
function updateReplay() {
  const ep = replay.ep;
  replay.trail = ep.frames.slice(0, replay.i + 1).map(q => linkPoints(q)[3]);
  $("scrub").value = replay.i;
  $("replayInfo").textContent = `episode ${ep.id} · ${ep.model} · ${ep.delay_ms} ms delay · t = ${(replay.i * 0.04).toFixed(2)} s`;
  $("play").textContent = replay.playing ? "Pause" : "Play";
  $("status").innerHTML = `Replaying episode ${ep.id}: ${ep.success ? '<span class="ok">reached</span>' : '<span class="bad">missed</span>'}` +
    `, final error ${ep.final_err_mm} mm`;
}
function stopReplay() { replay = null; $("replayBar").hidden = true; trail = []; }
$("scrub").oninput = () => { replay.i = +$("scrub").value; replay.playing = false; updateReplay(); };
$("play").onclick = () => { if (replay.i >= replay.ep.frames.length - 1) replay.i = 0; replay.playing = !replay.playing; updateReplay(); };
$("live").onclick = stopReplay;

connect();
setInterval(refreshEpisodes, 1500);
requestAnimationFrame(frame);
