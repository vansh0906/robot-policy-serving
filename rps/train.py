"""Research to production: train each policy version, export it to ONNX, check parity, run the release gate and
write a model card. Only models that pass the gate are marked deployable.

  reach-v1  the research handoff: one command per reply, trained on the lab's narrow target range
  reach-v2  same network, retrained on the full workspace after the customer edge-of-workspace report
  reach-v3  predicts a 25-step (0.5 s) action chunk so robots can ride out network delay
"""
import json
import os

import numpy as np

from . import data, evaluate
from .onnx_export import export
from .policy import MLP
from .runtime import sha256

VERSIONS = [
    dict(name="reach-v1", K=1, dist="NARROW", noise=0.0, rated_rtt_ms=100,
         note="Research handoff. One command per reply; trained on the lab's narrow target range."),
    dict(name="reach-v2", K=1, dist="FULL", noise=0.3, rated_rtt_ms=100,
         note="Retrained on the full reachable workspace (customer report: misses near the workspace edge)."),
    dict(name="reach-v3", K=25, dist="FULL", noise=0.3, rated_rtt_ms=300,
         note="Predicts a 0.5 s action chunk; robots skip the part already spent in flight (customer report: "
              "oscillation on a slow cellular link)."),
]
GATE_SUCCESS = 0.95


def gate(mlp, K, rated_rtt_ms, n_tasks=300):
    s, g = evaluate.tasks(n_tasks, seed=123)
    mode = "chunk" if K > 1 else "hold"
    out = {}
    for rtt in (0, rated_rtt_ms):
        r = evaluate.run(mlp.forward, s, g, K=K, mode=mode, rtt_ms=rtt, jitter_ms=0.25 * rtt)
        out[f"rtt_{rtt}ms"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "ok"}
    out["passed"] = all(v["success"] >= GATE_SUCCESS for v in out.values() if isinstance(v, dict))
    return out


def build(model_dir, episodes=800, epochs=30, hidden=256, versions=VERSIONS, log=print):
    import onnxruntime as ort
    os.makedirs(model_dir, exist_ok=True)
    cards = []
    for v in versions:
        X, Y = data.demos(episodes, K=v["K"], noise=v["noise"], dist=getattr(data, v["dist"]), seed=1)
        losses = []
        mlp = MLP([X.shape[1], hidden, hidden, Y.shape[1]]).fit(X, Y, epochs=epochs, log=lambda e, l: losses.append(l))
        npz = os.path.join(model_dir, v["name"] + ".npz")
        mlp.save(npz, {})
        os.remove(os.path.splitext(npz)[0] + ".json")
        onnx_path = export(mlp, os.path.join(model_dir, v["name"] + ".onnx"))
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        Xt = X[np.random.default_rng(5).choice(len(X), 2048, replace=False)]
        parity = float(np.abs(sess.run(None, {"obs": Xt})[0] - mlp.forward(Xt)).max())
        g = gate(mlp, v["K"], v["rated_rtt_ms"])
        card = dict(name=v["name"], K=v["K"], horizon_s=round(v["K"] * 0.02, 2), obs_dim=int(X.shape[1]),
                    action_dim=int(Y.shape[1]), note=v["note"],
                    training=dict(episodes=episodes, samples=int(len(X)), target_range=v["dist"].lower(),
                                  dart_noise=v["noise"], epochs=epochs, hidden=[hidden, hidden],
                                  final_mse=round(losses[-1], 6)),
                    onnx=dict(ir_version=8, opset=17, parity_max_abs=parity, sha256=None),
                    rated_rtt_ms=v["rated_rtt_ms"], gate=g, deployable=bool(g["passed"] and parity < 1e-4))
        card["sha256"] = sha256(onnx_path)
        card["onnx"]["sha256"] = card["sha256"]
        with open(os.path.join(model_dir, v["name"] + ".json"), "w") as f:
            json.dump(card, f, indent=2)
        cards.append(card)
        log(f"  {v['name']}: K={v['K']} parity {parity:.1e}  gate " +
            ", ".join(f"{k} {d['success']:.3f}" for k, d in g.items() if isinstance(d, dict)) +
            f"  -> {'deployable' if card['deployable'] else 'blocked'}")
    return cards
