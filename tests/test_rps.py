import asyncio
import json
import os
import sys
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rps import arm, evaluate, ws  # noqa: E402
from rps.client import run_robot  # noqa: E402
from rps.onnx_export import export  # noqa: E402
from rps.policy import MLP  # noqa: E402
from rps.runtime import ActionBuffer, Registry, ValidationError, check_state, sha256, validate  # noqa: E402
from rps.server import Server  # noqa: E402

MODELS = os.path.join(ROOT, "models")


def run(coro):
    return asyncio.run(coro)


# ---------- kinematics and expert ----------

def test_kinematics_consistent():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(arm.Q_LO, arm.Q_HI)
        e = 1e-6
        Jn = np.stack([(arm.fk(q + e * np.eye(3)[i]) - arm.fk(q - e * np.eye(3)[i])) / (2 * e) for i in range(3)], 1)
        assert np.abs(arm.jacobian(q) - Jn).max() < 1e-7
        assert np.allclose(arm.link_points(q)[3], arm.fk(q))
    for _ in range(20):
        p = arm.sample_target(rng)
        assert np.abs(arm.fk(arm.ik(p)) - p).max() < 1e-9
    P = rng.uniform(-0.6, 0.6, (200, 3))
    qb = arm.ik_batch(P)
    for p, qq in zip(P, qb):
        single = arm.ik(p)
        assert (single is None) == np.isnan(qq).any()


def test_expert_reaches():
    s, g = evaluate.tasks(150, seed=9)
    q, qd = s.copy(), np.zeros_like(s)
    for _ in range(int(arm.EPISODE_S / arm.DT_CTRL)):
        q, qd, _ = arm.step_batch(q, qd, arm.expert_batch(q, g))
    assert np.mean(np.linalg.norm(arm.fk(q) - g, axis=1) < arm.TOL) >= 0.99


# ---------- research to production ----------

def test_onnx_export_matches_numpy(tmp_path):
    import onnxruntime as ort
    m = MLP([18, 32, 32, 6], rng=np.random.default_rng(1))
    m.mu = np.random.default_rng(2).normal(size=18).astype(np.float32)
    m.sd = np.random.default_rng(3).uniform(0.5, 2, 18).astype(np.float32)
    path = export(m, str(tmp_path / "m.onnx"))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(4).normal(size=(37, 18)).astype(np.float32)   # dynamic batch dimension
    assert np.abs(sess.run(None, {"obs": x})[0] - m.forward(x)).max() < 1e-5


def test_registry_cards_and_gate():
    reg = Registry(MODELS)
    assert set(reg.names()) == {"reach-v1", "reach-v2", "reach-v3"}
    for n in reg.names():
        c = reg.card(n)
        assert c["sha256"] == sha256(os.path.join(MODELS, n + ".onnx"))
        assert c["onnx"]["parity_max_abs"] < 1e-4
    assert not reg.card("reach-v1")["deployable"]          # failed the gate on the full workspace
    assert reg.card("reach-v2")["deployable"] and reg.card("reach-v3")["deployable"]
    assert reg.default() == "reach-v3"
    obs = validate({"q": arm.Q_HOME.tolist(), "qd": [0, 0, 0], "target": [0.3, 0.1, 0.3]})[None]
    assert reg.run("reach-v3", obs).shape == (1, 75)


# ---------- validation and execution ----------

def test_state_validation():
    ok = {"q": [0.1, 0.8, -1.5], "qd": [0, 0, 0], "target": [0.3, 0.2, 0.3]}
    v = np.array(check_state(ok))
    assert np.allclose(validate(ok), arm.observation(v[:3], v[3:6], v[6:]).astype(np.float32))
    cases = [({**ok, "q": [0.1, float("nan"), -1.5]}, "non_finite"),
             ({**ok, "q": [5.7, 45.8, -85.9]}, "q_out_of_range"),
             ({**ok, "qd": [0, 0]}, "bad_shape"),
             ({"q": [0, 0, 0]}, "bad_shape"),
             ({**ok, "target": [30.0, 20.0, 30.0]}, "target_unreachable"),
             ({**ok, "qd": [9.0, 0, 0]}, "qd_out_of_range")]
    for state, code in cases:
        with pytest.raises(ValidationError) as e:
            check_state(state)
        assert e.value.code == code
    with pytest.raises(ValidationError) as e:
        check_state({**ok, "q": [5.7, 45.8, -85.9]})
    assert "degrees" in e.value.msg


def test_action_buffer_drops_stale_and_skips_time_in_flight():
    b = ActionBuffer(K=5)
    chunk = np.arange(15).reshape(5, 3) / 100
    assert b.on_reply(seq=10, obs_step=10, actions=chunk.ravel())
    assert not b.on_reply(seq=9, obs_step=9, actions=np.ones(15))      # arrived late, out of order
    assert b.stale == 1
    assert np.allclose(b.command(12), chunk[2])                           # 2 steps already spent in flight
    assert np.allclose(b.command(15), 0)                                  # past the chunk: stop
    h = ActionBuffer(K=1)
    h.on_reply(0, 0, [0.1, 0.2, 0.3])
    assert np.allclose(h.command(40), [0.1, 0.2, 0.3])                    # single-step model holds


def test_runtime_buffer_matches_batch_evaluator():
    """The robot-side ActionBuffer and the vectorised evaluator implement the same execution rule."""
    m = MLP.load(os.path.join(MODELS, "reach-v3.npz"))
    s, g = evaluate.tasks(3, seed=77)
    d = 7                                                                 # 140-159 ms RTT -> 7 control steps
    ref = evaluate.run(m.forward, s, g, K=25, mode="chunk", rtt_ms=150, record=True)["traj"]
    for i in range(3):
        sim, buf, inflight = arm.Arm(q=s[i].copy()), ActionBuffer(25), {}
        for t in range(ref.shape[0]):
            inflight[t + d] = (t, m.forward(arm.observation(sim.q, sim.qd, g[i])[None])[0])
            if t in inflight:
                st, a = inflight.pop(t)
                buf.on_reply(st, st, a)
            sim.step(buf.command(t))
        assert np.abs(sim.q - ref[-1, i]).max() < 1e-4


def test_chunking_beats_hold_under_delay():
    v2 = MLP.load(os.path.join(MODELS, "reach-v2.npz"))
    v3 = MLP.load(os.path.join(MODELS, "reach-v3.npz"))
    s, g = evaluate.tasks(60, seed=5)
    hold = evaluate.run(v2.forward, s, g, K=1, mode="hold", rtt_ms=300, jitter_ms=75)["success"]
    chunk = evaluate.run(v3.forward, s, g, K=25, mode="chunk", rtt_ms=300, jitter_ms=75)["success"]
    assert chunk >= 0.95 and hold <= 0.3


# ---------- WebSocket layer ----------

def test_websocket_roundtrip_sizes():
    async def main():
        async def handler(reader, writer):
            line = await reader.readline()
            headers = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b""):
                    break
                k, _, v = h.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            sock = await ws.server_handshake(reader, writer, headers)
            try:
                while True:
                    await sock.send(await sock.recv())
            except ws.ConnectionClosed:
                pass

        srv = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        c = await ws.connect("127.0.0.1", port, "/")
        for n in (0, 1, 125, 126, 65535, 70000):
            msg = "x" * n
            await c.send(msg)
            assert await c.recv() == msg
        await c.send(b"\x00\x01binary")
        assert await c.recv() == b"\x00\x01binary"
        await c.close()
        srv.close()
    run(main())


# ---------- server ----------

def http(port, method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_robot_protocol_and_faults():
    async def main():
        srv = await Server().start(port=0)
        c = await ws.connect("127.0.0.1", srv.port, "/ws/robot")
        hello = json.loads(await c.recv())
        assert hello["model"] == "reach-v3" and hello["K"] == 25
        good = {"type": "state", "seq": 1, "step": 1, "q": arm.Q_HOME.tolist(), "qd": [0, 0, 0], "target": [0.3, 0.2, 0.3]}
        await c.send(json.dumps(good))
        r = json.loads(await c.recv())
        assert r["type"] == "act" and r["seq"] == 1 and len(r["actions"]) == 75
        await c.send(json.dumps({**good, "seq": 2, "q": [0.1, float("nan"), -1.5]}))
        r = json.loads(await c.recv())
        assert r["type"] == "reject" and r["code"] == "non_finite"
        await c.send(json.dumps({**good, "seq": 3, "q": np.degrees(arm.Q_HOME).tolist()}))
        r = json.loads(await c.recv())
        assert r["code"] == "q_out_of_range" and "degrees" in r["msg"]
        await c.send(json.dumps({**good, "seq": 4}))                       # server still serving after bad input
        assert json.loads(await c.recv())["type"] == "act"
        await c.close()
        blocked = await ws.connect("127.0.0.1", srv.port, "/ws/robot?model=reach-v1")
        assert json.loads(await blocked.recv())["code"] == "not_deployable"
        await srv.stop()
    run(main())


def test_server_batches_concurrent_requests_correctly():
    async def main():
        srv = await Server(window_ms=5.0, max_batch=64).start(port=0)
        rng = np.random.default_rng(0)
        states = [[*arm.sample_start(rng), 0, 0, 0, *arm.sample_target(rng)] for _ in range(24)]
        outs = await asyncio.gather(*[srv.batcher.infer("reach-v3", s) for s in states])
        S = np.array(states)
        ref = srv.registry.run("reach-v3", arm.observation_batch(S[:, :3], S[:, 3:6], S[:, 6:]).astype(np.float32))
        assert np.abs(np.array(outs) - ref).max() < 1e-6
        assert max(srv.metrics.series["batch_size"]) > 1
        await srv.stop()
    run(main())


def test_http_api_and_static_files():
    async def main():
        srv = await Server(allow_experimental=False).start(port=0)
        loop = asyncio.get_running_loop()
        call = lambda *a: loop.run_in_executor(None, http, srv.port, *a)
        code, body = await call("GET", "/api/health")
        assert code == 200 and json.loads(body)["ok"]
        code, body = await call("GET", "/api/models")
        assert [c["name"] for c in json.loads(body)] == ["reach-v1", "reach-v2", "reach-v3"]
        code, _ = await call("POST", "/api/robots", {"model": "reach-v1"})
        assert code == 403
        code, body = await call("POST", "/api/robots", {"model": "reach-v3", "auto": False})
        assert code == 201
        rid = json.loads(body)["id"]
        code, body = await call("POST", f"/api/robots/{rid}/target", {"target": [3.0, 0, 0]})
        assert code == 400 and json.loads(body)["error"] == "target_unreachable"
        code, _ = await call("POST", f"/api/robots/{rid}/target", {"target": [0.35, -0.2, 0.25]})
        assert code == 200
        await asyncio.sleep(3.0)
        code, body = await call("GET", "/api/episodes")
        eps = json.loads(body)
        assert eps and eps[0]["success"] and eps[0]["final_err_mm"] < 15
        code, body = await call("GET", "/")
        assert code == 200 and b"Policy Serving Console" in body
        code, _ = await call("GET", "/web/../rps/server.py")
        assert code == 404
        code, _ = await call("POST", f"/api/robots/{rid}/config", {"delay_ms": -5})
        assert code == 400
        await srv.stop()
    run(main())


def test_remote_robot_reaches_through_delayed_link():
    async def main():
        srv = await Server().start(port=0)
        out = await run_robot("127.0.0.1", srv.port, "reach-v3", targets=2, delay_ms=200, jitter_ms=50, seed=11)
        await srv.stop()
        return out
    out = run(main())
    assert out["success"] == 1.0 and out["rtt_p50"] > 200
