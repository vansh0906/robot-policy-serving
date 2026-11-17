"""Policy server: HTTP API + WebSocket inference for robots + live state for browser viewers, on one asyncio loop.

  /ws/robot    robots stream their state at 50 Hz and get back commands (one, or a chunk of K)
  /ws/viewer   browsers get every robot's state at 20 Hz and can move targets
  /api/...     models, robots, metrics, episodes (see README)
  /            the 3-D viewer (web/)

Requests from all robots go through one dynamic batcher: it takes every request already queued (up to `max_batch`,
optionally waiting `window_ms` for more) and runs one ONNX Runtime call per model per batch, in a worker thread so
the event loop keeps reading sockets.
"""
import argparse
import asyncio
import collections
import concurrent.futures
import itertools
import json
import mimetypes
import os
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import arm, ws
from .runtime import ActionBuffer, Registry, ValidationError, check_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(values, p):
    return round(float(np.percentile(values, p)), 3) if len(values) else None


class Metrics:
    def __init__(self, keep=20000):
        self.series = collections.defaultdict(lambda: collections.deque(maxlen=keep))
        self.count = collections.Counter()

    def add(self, name, v):
        self.series[name].append(v)

    def summary(self):
        out = {"counters": dict(self.count)}
        for k, d in self.series.items():
            v = np.array(d)
            out[k] = {"n": len(v), "mean": round(float(v.mean()), 3), "p50": pct(v, 50), "p99": pct(v, 99),
                      "max": round(float(v.max()), 3)}
        return out


class Batcher:
    def __init__(self, registry, metrics, window_ms=0.0, max_batch=256):
        self.registry, self.metrics = registry, metrics
        self.window = window_ms / 1e3
        self.max_batch = max_batch
        self.queue = asyncio.Queue()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def infer(self, model, obs):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((model, obs, fut, time.perf_counter()))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = loop.time() + self.window
            while len(items) < self.max_batch:
                try:
                    items.append(self.queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                left = deadline - loop.time()
                if left <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), left))
                except asyncio.TimeoutError:
                    break
            groups = collections.defaultdict(list)
            for it in items:
                groups[it[0]].append(it)
            for model, group in groups.items():
                S = np.array([g[1] for g in group])                 # raw states -> features, vectorised
                X = arm.observation_batch(S[:, :3], S[:, 3:6], S[:, 6:]).astype(np.float32)
                t0 = time.perf_counter()
                try:
                    Y = await loop.run_in_executor(self.pool, self.registry.run, model, X)
                except Exception as e:                         # a bad model must not take the server down
                    for g in group:
                        if not g[2].done():
                            g[2].set_exception(e)
                    continue
                t1 = time.perf_counter()
                self.metrics.add("infer_ms", (t1 - t0) * 1e3)
                self.metrics.add("batch_size", len(group))
                self.metrics.count["batches"] += 1
                for g, y in zip(group, Y):
                    self.metrics.add("queue_ms", (t0 - g[3]) * 1e3)
                    if not g[2].done():
                        g[2].set_result(y)


class Robot:
    def __init__(self, rid, kind, model, K):
        self.id, self.kind, self.model, self.K = rid, kind, model, K
        self.q = arm.Q_HOME.copy()
        self.target = arm.fk(arm.Q_HOME)
        self.rtt = collections.deque(maxlen=500)
        self.delay_ms = 0.0
        self.jitter_ms = 0.0
        self.auto = False
        self.successes = 0
        self.timeouts = 0
        self.rejects = 0
        self.last_error = None
        self.status = "idle"
        self.ws = None
        self.task = None

    def snapshot(self):
        return {"id": self.id, "kind": self.kind, "model": self.model, "K": self.K, "status": self.status,
                "q": [round(float(x), 4) for x in self.q], "target": [round(float(x), 4) for x in self.target],
                "tip": [round(float(x), 4) for x in arm.fk(self.q)],
                "rtt_p50": pct(self.rtt, 50), "rtt_p99": pct(self.rtt, 99),
                "delay_ms": self.delay_ms, "jitter_ms": self.jitter_ms, "auto": self.auto,
                "successes": self.successes, "timeouts": self.timeouts, "rejects": self.rejects,
                "last_error": self.last_error}


class Server:
    def __init__(self, model_dir=None, window_ms=0.0, max_batch=256, viewer_hz=20, allow_experimental=False,
                 web_dir=None, episode_timeout_s=6.0):
        self.registry = Registry(model_dir or os.path.join(ROOT, "models"))
        self.metrics = Metrics()
        self.window_ms, self.max_batch = window_ms, max_batch
        self.viewer_hz = viewer_hz
        self.allow_experimental = allow_experimental
        self.web_dir = web_dir or os.path.join(ROOT, "web")
        self.episode_timeout_s = episode_timeout_s
        self.robots = {}
        self.episodes = collections.deque(maxlen=50)
        self.ids = itertools.count(1)
        self.ep_ids = itertools.count(1)
        self.viewers = set()
        self.rng = np.random.default_rng(int(time.time()))

    # ---------- lifecycle ----------
    async def start(self, host="127.0.0.1", port=8080):
        self.batcher = Batcher(self.registry, self.metrics, self.window_ms, self.max_batch)
        self._batch_task = asyncio.create_task(self.batcher.run())
        self.server = await asyncio.start_server(self.handle, host, port)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def stop(self):
        for r in list(self.robots.values()):
            if r.task:
                r.task.cancel()
        self._batch_task.cancel()
        self.server.close()
        await self.server.wait_closed()
        self.batcher.pool.shutdown(wait=False)

    # ---------- model selection ----------
    def check_model(self, name):
        name = name or self.registry.default()
        if name not in self.registry.models:
            raise ValidationError("unknown_model", f"no model named {name!r}; see /api/models")
        if not self.registry.card(name).get("deployable") and not self.allow_experimental:
            raise ValidationError("not_deployable", f"{name} has not passed the release gate (start the server "
                                  "with --allow-experimental to try it)")
        return name, int(self.registry.card(name)["K"])

    # ---------- one inference request ----------
    async def act(self, robot, state):
        t0 = time.perf_counter()
        self.metrics.count["requests"] += 1
        try:
            obs = check_state(state)
        except ValidationError as e:
            self.metrics.count["rejected"] += 1
            robot.rejects += 1
            robot.last_error = e.code
            return {"type": "reject", "seq": state.get("seq"), "code": e.code, "msg": e.msg}
        model = robot.model
        y = await self.batcher.infer(model, obs)
        self.metrics.add("server_ms", (time.perf_counter() - t0) * 1e3)
        return {"type": "act", "seq": state.get("seq"), "step": state.get("step"), "model": model, "K": robot.K,
                "actions": np.round(y.astype(float), 4).tolist()}

    # ---------- server-hosted simulated robots ----------
    def new_episode(self, robot, target):
        robot.target = np.asarray(target, float)
        robot.episode = {"id": next(self.ep_ids), "robot": robot.id, "model": robot.model, "target": robot.target.tolist(),
                         "delay_ms": robot.delay_ms, "start_q": robot.q.tolist(), "frames": [], "t0": time.time()}
        robot.status = "moving"

    def finish_episode(self, robot, ok, t):
        ep = robot.episode
        ep.update(success=ok, duration_s=round(t, 2), rtt_p50=pct(robot.rtt, 50), final_err_mm=round(float(
            np.linalg.norm(arm.fk(robot.q) - robot.target) * 1e3), 1))
        self.episodes.append(ep)
        robot.episode = None
        robot.status = "reached" if ok else "timeout"
        if ok:
            robot.successes += 1
        else:
            robot.timeouts += 1

    async def sim_loop(self, robot):
        sim = arm.Arm(q=robot.q.copy())
        buf = ActionBuffer(robot.K)
        loop = asyncio.get_running_loop()
        seq, step, hold, ep_step, idle_until = 0, 0, 0, 0, None
        robot.episode = None
        t_next = loop.time()

        async def request(seq, step, state):
            sent = loop.time()
            d = robot.delay_ms + (self.rng.exponential(robot.jitter_ms) if robot.jitter_ms > 0 else 0.0)
            if d > 0:
                await asyncio.sleep(d / 2e3)
            reply = await self.act(robot, state)
            if d > 0:
                await asyncio.sleep(d / 2e3)
            if reply["type"] == "act" and buf.K == robot.K:
                buf.on_reply(seq, step, reply["actions"])
                robot.rtt.append((loop.time() - sent) * 1e3)

        while True:
            if buf.K != robot.K:                      # model was hot-swapped
                buf = ActionBuffer(robot.K)
            if robot.episode is None and robot.auto and (idle_until is None or loop.time() >= idle_until):
                self.new_episode(robot, arm.sample_target(self.rng))
                ep_step, hold = 0, 0
            state = {"seq": seq, "step": step, "q": sim.q.tolist(), "qd": sim.qd.tolist(), "target": robot.target.tolist()}
            asyncio.create_task(request(seq, step, state))
            sim.step(buf.command(step))
            robot.q = sim.q.copy()
            seq += 1
            step += 1
            if robot.episode is not None:
                ep_step += 1
                if ep_step % 2 == 0:
                    robot.episode["frames"].append([round(float(x), 4) for x in sim.q])
                err = np.linalg.norm(arm.fk(sim.q) - robot.target)
                hold = hold + 1 if err < arm.TOL else 0
                t = ep_step * arm.DT_CTRL
                if hold * arm.DT_CTRL >= arm.HOLD_S:
                    self.finish_episode(robot, True, t)
                    idle_until = loop.time() + 1.0
                elif t >= self.episode_timeout_s:
                    self.finish_episode(robot, False, t)
                    idle_until = loop.time() + 1.0
            t_next += arm.DT_CTRL
            await asyncio.sleep(max(0.0, t_next - loop.time()))

    def add_sim(self, model=None, delay_ms=0.0, jitter_ms=0.0, auto=True):
        name, K = self.check_model(model)
        r = Robot(f"sim-{next(self.ids)}", "sim", name, K)
        r.delay_ms, r.jitter_ms, r.auto = float(delay_ms), float(jitter_ms), bool(auto)
        r.episode = None
        self.robots[r.id] = r
        r.task = asyncio.create_task(self.sim_loop(r))
        return r

    def set_target(self, robot, target):
        t = np.asarray(target, dtype=float)
        if t.shape != (3,) or not np.isfinite(t).all() or arm.ik(t) is None:
            raise ValidationError("target_unreachable", "target is outside the arm's reachable workspace")
        if robot.kind == "sim":
            if robot.episode is not None:
                self.finish_episode(robot, False, len(robot.episode["frames"]) * 2 * arm.DT_CTRL)
            robot.auto = False
            self.new_episode(robot, t)
        else:
            robot.target = t
            if robot.ws:
                asyncio.create_task(robot.ws.send(json.dumps({"type": "set_target", "target": t.tolist()})))

    # ---------- connection handling ----------
    async def handle(self, reader, writer):
        try:
            line = await asyncio.wait_for(reader.readline(), 10)
            if not line:
                return
            method, target, _ = line.decode("latin-1").split(" ", 2)
            headers = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            url = urlparse(target)
            if headers.get("upgrade", "").lower() == "websocket":
                sock = await ws.server_handshake(reader, writer, headers)
                if sock is None:
                    return
                s = writer.get_extra_info("socket")
                if s is not None:
                    import socket
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if url.path == "/ws/robot":
                    await self.robot_session(sock, parse_qs(url.query))
                elif url.path == "/ws/viewer":
                    await self.viewer_session(sock)
                else:
                    await sock.close(1008)
                return
            n = int(headers.get("content-length", 0) or 0)
            if n > 65536:
                return await self.respond(writer, 413, {"error": "body too large"})
            body = await reader.readexactly(n) if n else b""
            await self.http(writer, method, url, body)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.TimeoutError, ValueError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def respond(self, writer, code, obj=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else json.dumps(obj).encode()
        reason = {200: "OK", 201: "Created", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
                  405: "Method Not Allowed", 413: "Payload Too Large"}.get(code, "OK")
        writer.write(f"HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
                     "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode() + data)
        await writer.drain()

    async def http(self, writer, method, url, body):
        path = url.path.rstrip("/") or "/"
        parts = path.strip("/").split("/")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return await self.respond(writer, 400, {"error": "body is not JSON"})
        try:
            if path == "/" or (parts[0] == "web" and method == "GET"):
                return await self.static(writer, "index.html" if path == "/" else "/".join(parts[1:]))
            if path == "/api/health":
                return await self.respond(writer, 200, {"ok": True, "models": self.registry.names(),
                                                        "robots": len(self.robots)})
            if path == "/api/models":
                return await self.respond(writer, 200, [self.registry.card(n) for n in self.registry.names()])
            if path == "/api/metrics":
                m = self.metrics.summary()
                m["robots"] = len(self.robots)
                m["viewers"] = len(self.viewers)
                m["batching"] = {"window_ms": self.window_ms, "max_batch": self.max_batch}
                return await self.respond(writer, 200, m)
            if path == "/api/episodes":
                return await self.respond(writer, 200, [{k: v for k, v in e.items() if k != "frames"}
                                                        for e in reversed(self.episodes)])
            if len(parts) == 3 and parts[:2] == ["api", "episodes"]:
                ep = next((e for e in self.episodes if str(e["id"]) == parts[2]), None)
                return await self.respond(writer, 200 if ep else 404, ep or {"error": "no such episode"})
            if path == "/api/robots" and method == "GET":
                return await self.respond(writer, 200, [r.snapshot() for r in self.robots.values()])
            if path == "/api/robots" and method == "POST":
                r = self.add_sim(data.get("model"), data.get("delay_ms", 0), data.get("jitter_ms", 0),
                                 data.get("auto", True))
                return await self.respond(writer, 201, r.snapshot())
            if len(parts) >= 3 and parts[:2] == ["api", "robots"]:
                r = self.robots.get(parts[2])
                if r is None:
                    return await self.respond(writer, 404, {"error": "no such robot"})
                if len(parts) == 3 and method == "DELETE":
                    if r.task:
                        r.task.cancel()
                    self.robots.pop(r.id, None)
                    return await self.respond(writer, 200, {"deleted": r.id})
                if len(parts) == 4 and parts[3] == "target" and method == "POST":
                    self.set_target(r, data.get("target"))
                    return await self.respond(writer, 200, r.snapshot())
                if len(parts) == 4 and parts[3] == "config" and method == "POST":
                    self.configure(r, data)
                    return await self.respond(writer, 200, r.snapshot())
            return await self.respond(writer, 404, {"error": "not found"})
        except ValidationError as e:
            return await self.respond(writer, 403 if e.code == "not_deployable" else 400,
                                      {"error": e.code, "msg": e.msg})

    def configure(self, r, data):
        if "model" in data:
            r.model, r.K = self.check_model(data["model"])
        for k in ("delay_ms", "jitter_ms"):
            if k in data:
                v = float(data[k])
                if not 0 <= v <= 2000:
                    raise ValidationError("bad_value", f"{k} must be between 0 and 2000")
                setattr(r, k, v)
        if "auto" in data:
            r.auto = bool(data["auto"])

    async def static(self, writer, rel):
        path = os.path.normpath(os.path.join(self.web_dir, rel))
        if not path.startswith(os.path.normpath(self.web_dir) + os.sep) or not os.path.isfile(path):
            return await self.respond(writer, 404, {"error": "not found"})
        with open(path, "rb") as f:
            raw = f.read()
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return await self.respond(writer, 200, raw=raw, ctype=ctype)

    # ---------- WebSocket sessions ----------
    async def robot_session(self, sock, query):
        try:
            name, K = self.check_model((query.get("model") or [None])[0])
        except ValidationError as e:
            await sock.send(json.dumps({"type": "error", "code": e.code, "msg": e.msg}))
            return await sock.close(1008)
        r = Robot((query.get("id") or [f"robot-{next(self.ids)}"])[0], "remote", name, K)
        r.ws = sock
        self.robots[r.id] = r
        await sock.send(json.dumps({"type": "hello", "robot": r.id, "model": name, "K": K}))
        pending = set()

        async def one(state):
            reply = await self.act(r, state)
            if reply["type"] == "act":
                r.q = np.asarray(state["q"], float)
                r.target = np.asarray(state["target"], float)
                r.status = "connected"
                if state.get("rtt_ms") is not None:
                    r.rtt.append(float(state["rtt_ms"]))
            try:
                await sock.send(json.dumps(reply))
            except ws.ConnectionClosed:
                pass

        try:
            while True:
                msg = json.loads(await sock.recv())
                if msg.get("type") == "state":
                    t = asyncio.create_task(one(msg))
                    pending.add(t)
                    t.add_done_callback(pending.discard)
                elif msg.get("type") == "config" and "model" in msg:
                    r.model, r.K = self.check_model(msg["model"])
                    await sock.send(json.dumps({"type": "hello", "robot": r.id, "model": r.model, "K": r.K}))
        except (ws.ConnectionClosed, json.JSONDecodeError, ValidationError):
            pass
        finally:
            self.robots.pop(r.id, None)
            await sock.close()

    async def viewer_session(self, sock):
        self.viewers.add(sock)
        await sock.send(json.dumps({"type": "hello", "models": [self.registry.card(n) for n in self.registry.names()],
                                    "arm": {"H0": arm.H0, "L1": arm.L1, "L2": arm.L2,
                                            "q_lo": arm.Q_LO.tolist(), "q_hi": arm.Q_HI.tolist()}}))

        async def pump():
            period = 1.0 / self.viewer_hz
            while True:
                snap = {"type": "state", "t": time.time(), "robots": [r.snapshot() for r in self.robots.values()]}
                try:
                    # a slow browser only delays its own updates; it can never stall robots or the batcher
                    await asyncio.wait_for(sock.send(json.dumps(snap)), 2.0)
                except asyncio.TimeoutError:
                    self.metrics.count["viewer_dropped_slow"] += 1
                    return
                await asyncio.sleep(period)

        task = asyncio.create_task(pump())
        try:
            while not task.done():
                msg = json.loads(await sock.recv())
                r = self.robots.get(msg.get("robot", ""))
                try:
                    if msg.get("type") == "set_target" and r:
                        self.set_target(r, msg.get("target"))
                    elif msg.get("type") == "config" and r:
                        self.configure(r, msg)
                except (ValidationError, ValueError, TypeError) as e:
                    await sock.send(json.dumps({"type": "error", "code": getattr(e, "code", "bad_value"),
                                                "msg": getattr(e, "msg", str(e))}))
        except (ws.ConnectionClosed, json.JSONDecodeError):
            pass
        finally:
            task.cancel()
            self.viewers.discard(sock)


async def main_async(args):
    srv = Server(args.models, args.window_ms, args.max_batch, allow_experimental=args.allow_experimental)
    await srv.start(args.host, args.port)
    for _ in range(args.sims):
        srv.add_sim(delay_ms=args.sim_delay_ms)
    print(f"policy server on http://{args.host}:{srv.port}  models: {', '.join(srv.registry.names())}", flush=True)
    await asyncio.Event().wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--window-ms", type=float, default=0.0, help="extra wait to fill a batch (0: take what is queued)")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--sims", type=int, default=1, help="simulated robots to start with")
    ap.add_argument("--sim-delay-ms", type=float, default=0.0)
    ap.add_argument("--allow-experimental", action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
