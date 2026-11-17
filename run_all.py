"""Runs every study and writes results/*.json, results/RESULTS.md and docs/img/results.png.

    python run_all.py              # uses the committed models (about 3 minutes)
    python run_all.py --retrain    # also retrains, exports and gates the three model versions (+2 minutes)
    python run_all.py --quick      # smaller load test, for a fast check
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from rps import data, evaluate  # noqa: E402
from rps.client import run_robot  # noqa: E402
from rps.policy import MLP  # noqa: E402
from rps.train import build  # noqa: E402

RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
RTTS = [0, 50, 100, 150, 200, 250, 300, 400, 500]


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)


def load(name):
    return MLP.load(os.path.join(MODELS, name + ".npz"))


def coverage_study(n=600):
    """Customer report 1: misses near the edge of the workspace. Split the customer task set by whether the target
    lies inside the research lab's training range."""
    s, g = evaluate.tasks(n, seed=321)
    r, z, yaw = np.hypot(g[:, 0], g[:, 1]), g[:, 2], np.arctan2(g[:, 1], g[:, 0])
    nr = data.NARROW
    inside = ((r >= nr["r_range"][0]) & (r <= nr["r_range"][1]) & (z >= nr["z_range"][0]) & (z <= nr["z_range"][1])
              & (yaw >= nr["yaw"][0]) & (yaw <= nr["yaw"][1]))
    out = {"tasks": n, "inside_lab_range": int(inside.sum()), "outside_lab_range": int((~inside).sum())}
    for name in ("reach-v1", "reach-v2"):
        ok = evaluate.run(load(name).forward, s, g, K=1, mode="hold")["ok"]
        out[name] = {"all": round(float(ok.mean()), 4), "inside": round(float(ok[inside].mean()), 4),
                     "outside": round(float(ok[~inside].mean()), 4)}
    return out


def latency_study(n=300):
    """Customer report 2: oscillation on a slow link. Success vs round-trip time (jitter = 25% of RTT)."""
    s, g = evaluate.tasks(n, seed=123)
    v2, v3 = load("reach-v2"), load("reach-v3")
    rows = []
    for rtt in RTTS:
        row = {"rtt_ms": rtt}
        for label, m, K, mode in (("v2_hold", v2, 1, "hold"), ("v3_chunk", v3, 25, "chunk"),
                                  ("v3_chunk_noskip", v3, 25, "chunk_noskip")):
            r = evaluate.run(m.forward, s, g, K=K, mode=mode, rtt_ms=rtt, jitter_ms=0.25 * rtt)
            row[label] = {"success": round(r["success"], 4), "t_median": r["t_median"] and round(r["t_median"], 2)}
        rows.append(row)
        print(f"  RTT {rtt:3d} ms: v2 hold {row['v2_hold']['success']:.3f}  v3 chunk {row['v3_chunk']['success']:.3f}  "
              f"v3 no-skip {row['v3_chunk_noskip']['success']:.3f}", flush=True)
    return rows


async def socket_check(port, model, robots=4, targets=5, delay_ms=150, jitter_ms=40):
    outs = await asyncio.gather(*[run_robot("127.0.0.1", port, model, targets, delay_ms, jitter_ms, seed=100 + i,
                                            robot_id=f"{model}-{i}") for i in range(robots)])
    res = [r for o in outs for r in o["results"]]
    return {"model": model, "robots": robots, "targets": len(res), "delay_ms": delay_ms, "jitter_ms": jitter_ms,
            "success": round(float(np.mean([r["success"] for r in res])), 4),
            "rtt_p50_ms": round(float(np.median([o["rtt_p50"] for o in outs])), 1),
            "stale_replies_dropped": int(sum(o["stale_dropped"] for o in outs))}


def end_to_end(port=8781):
    """The same comparison over real WebSockets: 4 robot clients per model, 150 ms delay + 40 ms jitter."""
    import loadtest
    srv = subprocess.Popen([sys.executable, "-m", "rps.server", "--port", str(port), "--sims", "0"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                loadtest.get(port, "/api/health")
                break
            except Exception:
                time.sleep(0.1)
        out = []
        for model in ("reach-v2", "reach-v3"):
            for delay in (150, 300):
                r = asyncio.run(socket_check(port, model, delay_ms=delay, jitter_ms=0.25 * delay))
                out.append(r)
                print(f"  sockets {model} {delay} ms: success {r['success']:.2f}  rtt p50 {r['rtt_p50_ms']} ms  "
                      f"stale dropped {r['stale_replies_dropped']}", flush=True)
        return out
    finally:
        srv.terminate()
        srv.wait()


def load_study(quick=False):
    import loadtest
    robots = [1, 50, 100] if quick else [1, 25, 50, 100, 150, 200]
    rows = []
    for mode in ("nobatch", "batch"):
        for n in robots:
            r = loadtest.run_case(mode, n, seconds=3.0 if quick else 5.0)
            rows.append(r)
            print(f"  load {mode:8s} {n:4d} robots: p50 {r['rtt_p50_ms']} ms  p99 {r['rtt_p99_ms']} ms  "
                  f"miss {r['deadline_miss_pct']}%  batch mean {r['batch_mean']}", flush=True)
    return rows


def plot(cov, lat, load_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = [r["rtt_ms"] for r in lat]
    for key, lab, st in (("v2_hold", "reach-v2: one command, hold last", "o-"),
                         ("v3_chunk", "reach-v3: 0.5 s chunk, time-indexed", "s-"),
                         ("v3_chunk_noskip", "reach-v3: chunk played from arrival", "^--")):
        ax[0].plot(x, [100 * r[key]["success"] for r in lat], st, label=lab)
    ax[0].axhline(95, color="gray", lw=0.8, ls=":")
    ax[0].set(xlabel="round-trip time (ms), jitter 25%", ylabel="reach success (%)", title="Policy vs network delay")
    ax[0].legend(fontsize=8)
    for mode, st in (("nobatch", "o-"), ("batch", "s-")):
        rr = [r for r in load_rows if r["mode"] == mode]
        ax[1].plot([r["robots"] for r in rr], [r["deadline_miss_pct"] for r in rr], st,
                   label={"nobatch": "one inference call per request", "batch": "dynamic batching"}[mode])
    ax[1].set(xlabel="robots at 50 Hz", ylabel="replies later than 20 ms (%)", title="Server load (real WebSockets)")
    ax[1].legend(fontsize=8)
    labels = ["inside lab range", "outside lab range"]
    w = 0.38
    for i, name in enumerate(("reach-v1", "reach-v2")):
        ax[2].bar(np.arange(2) + (i - 0.5) * w, [100 * cov[name]["inside"], 100 * cov[name]["outside"]], w, label=name)
    ax[2].set_xticks(range(2), labels)
    ax[2].set(ylabel="reach success (%)", title="Workspace coverage", ylim=(0, 105))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "docs", "img"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "docs", "img", "results.png"), dpi=130)


def report(cards, cov, lat, e2e, load_rows):
    L = ["# Results", "", "Generated by `python run_all.py`. Machine: the load test runs the server and the robots on "
         "the same 2-vCPU computer, so its absolute numbers are a floor, not a benchmark.", "",
         "## Model registry and release gate", "",
         "| Model | Chunk | Gate: success at 0 ms | at rated RTT | ONNX parity (max abs) | Deployable |", "|---|---|---|---|---|---|"]
    for c in cards:
        g = c["gate"]
        rated = [k for k in g if k.startswith("rtt_") and k != "rtt_0ms"][0]
        L.append(f"| {c['name']} | {c['K']} | {g['rtt_0ms']['success']:.3f} | {g[rated]['success']:.3f} ({rated[4:]}) "
                 f"| {c['onnx']['parity_max_abs']:.1e} | {'yes' if c['deployable'] else 'no'} |")
    L += ["", "## Workspace coverage (customer report 1)", "",
          f"{cov['tasks']} customer tasks; {cov['outside_lab_range']} have targets outside the lab's training range.", "",
          "| Model | All | Inside lab range | Outside |", "|---|---|---|---|"]
    for name in ("reach-v1", "reach-v2"):
        c = cov[name]
        L.append(f"| {name} | {c['all']:.3f} | {c['inside']:.3f} | {c['outside']:.3f} |")
    L += ["", "## Network delay (customer report 2)", "", "300 tasks, simulated time, RTT jitter 25%.", "",
          "| RTT (ms) | reach-v2 hold | reach-v3 chunk | reach-v3 chunk, no skip |", "|---|---|---|---|"]
    for r in lat:
        L.append(f"| {r['rtt_ms']} | {r['v2_hold']['success']:.3f} | {r['v3_chunk']['success']:.3f} | "
                 f"{r['v3_chunk_noskip']['success']:.3f} |")
    L += ["", "Same check over real WebSockets (4 robot clients x 5 targets per row):", "",
          "| Model | Delay | Success | RTT p50 (ms) | Stale replies dropped |", "|---|---|---|---|---|"]
    for r in e2e:
        L.append(f"| {r['model']} | {r['delay_ms']} ms + {r['jitter_ms']:.0f} ms jitter | {r['success']:.2f} | "
                 f"{r['rtt_p50_ms']} | {r['stale_replies_dropped']} |")
    L += ["", "## Server load", "", "| Mode | Robots | RTT p50 (ms) | RTT p99 (ms) | Late replies (>20 ms) | Mean batch |",
          "|---|---|---|---|---|---|"]
    for r in load_rows:
        L.append(f"| {r['mode']} | {r['robots']} | {r['rtt_p50_ms']} | {r['rtt_p99_ms']} | {r['deadline_miss_pct']}% | "
                 f"{r['batch_mean']} |")
    with open(os.path.join(RES, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    if a.retrain:
        print("Training, exporting and gating models")
        build(MODELS)
    cards = [json.load(open(os.path.join(MODELS, f"{n}.json"))) for n in ("reach-v1", "reach-v2", "reach-v3")]
    print("Coverage study")
    cov = coverage_study()
    print(f"  reach-v1 outside lab range {cov['reach-v1']['outside']:.3f}, reach-v2 {cov['reach-v2']['outside']:.3f}")
    print("Latency study")
    lat = latency_study()
    print("End-to-end over WebSockets")
    e2e = end_to_end()
    print("Load study")
    load_rows = load_study(a.quick)
    save("coverage.json", cov)
    save("latency.json", lat)
    save("end_to_end.json", e2e)
    save("load.json", load_rows)
    plot(cov, lat, load_rows)
    report(cards, cov, lat, e2e, load_rows)
    print(f"done in {time.time() - t0:.0f} s -> results/RESULTS.md, docs/img/results.png")


if __name__ == "__main__":
    main()
