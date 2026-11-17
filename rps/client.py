"""A robot client: runs the arm at 50 Hz and gets its commands from the policy server over a WebSocket.

    python -m rps.client --targets 5 --delay-ms 150

--delay-ms adds network delay (half on the way out, half on the way back). --fault injects a customer-style fault
(nan: one encoder reading turns NaN; degrees: joint angles sent in degrees) to check that the server rejects it
and the robot stops safely instead of moving.
"""
import argparse
import asyncio
import json
import time

import numpy as np

from . import arm, ws
from .runtime import ActionBuffer


async def run_robot(host, port, model=None, targets=3, delay_ms=0.0, jitter_ms=0.0, fault=None, fault_step=50,
                    seed=0, timeout_s=6.0, robot_id=None, log=None):
    rng = np.random.default_rng(seed)
    q = f"/ws/robot?model={model}" if model else "/ws/robot"
    if robot_id:
        q += ("&" if "?" in q else "?") + f"id={robot_id}"
    sock = await ws.connect(host, port, q)
    hello = json.loads(await sock.recv())
    if hello["type"] != "hello":
        await sock.close()
        raise RuntimeError(hello)
    K = hello["K"]
    buf = ActionBuffer(K)
    sim = arm.Arm(q=arm.sample_start(rng))
    loop = asyncio.get_running_loop()
    sent_at, rtts, rejects, results = {}, [], [], []
    state = {"target": arm.sample_target(rng), "last_rtt": None, "hello": hello}

    def net_delay():
        d = delay_ms + (rng.exponential(jitter_ms) if jitter_ms > 0 else 0.0)
        return d / 2e3

    async def deliver(msg, d):
        if d:
            await asyncio.sleep(d)                 # each reply is delayed on its own, like packets in flight
        handle(msg)

    async def receiver():
        while True:
            msg = json.loads(await sock.recv())
            d = net_delay()
            if d:
                asyncio.create_task(deliver(msg, d))
            else:
                handle(msg)

    def handle(msg):
        if msg["type"] == "act":
            sent = sent_at.pop(msg["seq"], None)
            if buf.on_reply(msg["seq"], msg["step"], msg["actions"]) and sent is not None:
                rtt = (loop.time() - sent) * 1e3
                rtts.append(rtt)
                state["last_rtt"] = rtt
        elif msg["type"] == "reject":
            sent_at.pop(msg.get("seq"), None)
            rejects.append(msg["code"])
            buf.chunk = None                       # no valid command: stop the arm
        elif msg["type"] == "set_target":
            state["target"] = np.asarray(msg["target"], float)

    async def send(msg):
        d = net_delay()
        if d:
            await asyncio.sleep(d)
        try:
            await sock.send(json.dumps(msg))
        except ws.ConnectionClosed:
            pass

    rx = asyncio.create_task(receiver())
    step, seq = 0, 0
    t_next = loop.time()
    try:
        for i in range(targets):
            if i > 0:
                state["target"] = arm.sample_target(rng)
            hold, t0_step, ok = 0, step, False
            while (step - t0_step) * arm.DT_CTRL < timeout_s:
                qs = sim.q.copy()
                if fault == "nan" and step == fault_step:
                    qs[1] = float("nan")
                if fault == "degrees" and step >= fault_step:
                    qs = np.degrees(qs)
                msg = {"type": "state", "seq": seq, "step": step, "q": [float(x) for x in qs],
                       "qd": sim.qd.tolist(), "target": state["target"].tolist(), "rtt_ms": state["last_rtt"]}
                sent_at[seq] = loop.time()
                asyncio.create_task(send(msg))
                sim.step(buf.command(step))
                step += 1
                seq += 1
                err = np.linalg.norm(arm.fk(sim.q) - state["target"])
                hold = hold + 1 if err < arm.TOL else 0
                if hold * arm.DT_CTRL >= arm.HOLD_S:
                    ok = True
                    break
                t_next += arm.DT_CTRL
                await asyncio.sleep(max(0.0, t_next - loop.time()))
            results.append({"success": ok, "time_s": round((step - t0_step) * arm.DT_CTRL, 2),
                            "final_err_mm": round(float(err * 1e3), 1)})
            if log:
                log(f"target {i + 1}: {'reached' if ok else 'missed'} in {results[-1]['time_s']} s")
    finally:
        rx.cancel()
        await sock.close()
    return {"robot": hello["robot"], "model": hello["model"], "K": K, "results": results,
            "success": float(np.mean([r["success"] for r in results])) if results else 0.0,
            "rtt_p50": float(np.percentile(rtts, 50)) if rtts else None,
            "rtt_p99": float(np.percentile(rtts, 99)) if rtts else None,
            "stale_dropped": buf.stale, "rejects": rejects, "q_final": sim.q.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=None)
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--fault", choices=["nan", "degrees"], default=None)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 1000)
    a = ap.parse_args()
    out = asyncio.run(run_robot(a.host, a.port, a.model, a.targets, a.delay_ms, a.jitter_ms, a.fault, seed=a.seed,
                                log=print))
    print(json.dumps({k: v for k, v in out.items() if k != "q_final"}, indent=2))


if __name__ == "__main__":
    main()
