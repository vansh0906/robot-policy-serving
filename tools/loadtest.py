"""Load test: N robots each send a state every 20 ms over real WebSockets; measure round-trip time and deadline misses.

The server runs in its own process; the robots run in this process (one asyncio task each, no physics, so the test
measures the serving path). A reply later than one control period (20 ms) is a deadline miss.

    python tools/loadtest.py --robots 1 10 50 100 200 --modes nobatch batch
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rps import arm, ws  # noqa: E402

MODES = {"nobatch": dict(window_ms=0.0, max_batch=1),
         "batch": dict(window_ms=0.0, max_batch=256),        # opportunistic: take whatever is already queued
         "batch_2ms": dict(window_ms=2.0, max_batch=256)}    # also wait up to 2 ms to fill the batch
DEADLINE_MS = 20.0


async def robot(port, model, seconds, warmup, rng, out):
    sock = await ws.connect("127.0.0.1", port, f"/ws/robot?model={model}")
    await sock.recv()
    loop = asyncio.get_running_loop()
    q = arm.sample_start(rng)
    tgt = arm.sample_target(rng)
    sent = {}
    rtts = []
    t_start = loop.time()

    async def rx():
        while True:
            m = json.loads(await sock.recv())
            t = sent.pop(m.get("seq"), None)
            if t is not None and t - t_start >= warmup:
                rtts.append((loop.time() - t) * 1e3)

    task = asyncio.create_task(rx())
    await asyncio.sleep(rng.uniform(0, arm.DT_CTRL))                 # robots are not phase-locked
    seq, t_next, n_sent = 0, loop.time(), 0
    while loop.time() - t_start < seconds + warmup:
        sent[seq] = loop.time()
        if sent[seq] - t_start >= warmup:
            n_sent += 1
        await sock.send(json.dumps({"type": "state", "seq": seq, "step": seq, "q": q.tolist(), "qd": [0.0, 0.0, 0.0],
                                    "target": tgt.tolist()}))
        seq += 1
        t_next += arm.DT_CTRL
        await asyncio.sleep(max(0.0, t_next - loop.time()))
    await asyncio.sleep(0.3)
    task.cancel()
    await sock.close()
    out.append((n_sent, rtts))


async def fleet(port, n, model, seconds, warmup):
    out = []
    rng = np.random.default_rng(n)
    await asyncio.gather(*[robot(port, model, seconds, warmup, np.random.default_rng(rng.integers(1 << 30)), out)
                           for _ in range(n)])
    sent = sum(o[0] for o in out)
    r = np.concatenate([np.array(o[1]) for o in out]) if out else np.array([])
    lost = sent - len(r)
    return {"robots": n, "offered_hz": n / arm.DT_CTRL, "replies": int(len(r)), "sent": int(sent),
            "rtt_p50_ms": round(float(np.percentile(r, 50)), 2) if len(r) else None,
            "rtt_p99_ms": round(float(np.percentile(r, 99)), 2) if len(r) else None,
            "deadline_miss_pct": round(100.0 * (np.sum(r > DEADLINE_MS) + max(lost, 0)) / max(sent, 1), 2)}


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return json.loads(r.read())


def run_case(mode, n, model="reach-v3", seconds=5.0, warmup=1.0, port=8790):
    cfg = MODES[mode]
    srv = subprocess.Popen([sys.executable, "-m", "rps.server", "--port", str(port), "--sims", "0",
                            "--window-ms", str(cfg["window_ms"]), "--max-batch", str(cfg["max_batch"])],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                get(port, "/api/health")
                break
            except Exception:
                time.sleep(0.1)
        res = asyncio.run(fleet(port, n, model, seconds, warmup))
        m = get(port, "/api/metrics")
        res.update(mode=mode, model=model, batch_mean=m.get("batch_size", {}).get("mean"),
                   batch_max=m.get("batch_size", {}).get("max"), infer_p50_ms=m.get("infer_ms", {}).get("p50"),
                   server_p99_ms=m.get("server_ms", {}).get("p99"))
        return res
    finally:
        srv.terminate()
        srv.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", type=int, nargs="+", default=[1, 10, 50, 100, 200])
    ap.add_argument("--modes", nargs="+", default=["nobatch", "batch"])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--model", default="reach-v3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for mode in a.modes:
        for n in a.robots:
            r = run_case(mode, n, a.model, a.seconds)
            rows.append(r)
            print(f"{mode:9s} {n:4d} robots: p50 {r['rtt_p50_ms']} ms  p99 {r['rtt_p99_ms']} ms  "
                  f"miss {r['deadline_miss_pct']}%  batch mean {r['batch_mean']}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
