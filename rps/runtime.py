"""Pieces shared by the server and the robot client: input validation, the model registry and the action buffer."""
import glob
import hashlib
import json
import math
import os

import numpy as np

from . import arm


class ValidationError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


_LO = [float(x) - 0.2 for x in arm.Q_LO]
_HI = [float(x) + 0.2 for x in arm.Q_HI]
_REACH = arm.L1 + arm.L2 + 0.02


def check_state(state):
    """Check a robot's raw state (SI units) in plain Python (this runs for every request) and return the 9 numbers
    q, qd, target. Raises ValidationError with a code the robot can act on."""
    try:
        q, qd, tgt = state["q"], state["qd"], state["target"]
        if len(q) != 3 or len(qd) != 3 or len(tgt) != 3:
            raise ValidationError("bad_shape", "q, qd and target must have 3 numbers each")
        vals = [float(v) for v in (*q, *qd, *tgt)]
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError):
        raise ValidationError("bad_shape", "state needs q, qd and target, 3 numbers each") from None
    if not all(math.isfinite(v) for v in vals):
        raise ValidationError("non_finite", "state contains NaN or Inf (check the encoder driver)")
    if any(not (lo <= v <= hi) for v, lo, hi in zip(vals[:3], _LO, _HI)):
        hint = " - values look like degrees; send radians" if max(abs(v) for v in vals[:3]) > 2 * math.pi else ""
        raise ValidationError("q_out_of_range", f"joint angles outside the arm's range{hint}")
    if any(abs(v) > 3 * arm.V_MAX for v in vals[3:6]):
        raise ValidationError("qd_out_of_range", "joint speeds above 3x the rated limit (check units: rad/s)")
    x, y, z = vals[6:]
    if math.hypot(math.hypot(x, y), z - arm.H0) > _REACH or z < -0.02:
        raise ValidationError("target_unreachable", "target is outside the arm's workspace (check units: metres)")
    return vals


def validate(state):
    """check_state + the policy observation, built on the server so features always match training."""
    v = np.array(check_state(state))
    return arm.observation(v[:3], v[3:6], v[6:]).astype(np.float32)


class ActionBuffer:
    """Executes replies on the robot. Replies older than the newest one received are dropped (they can arrive out of
    order after a network hiccup). Chunked models: the command for the current step is element (step - obs_step) of
    the newest chunk, so time already spent in flight is skipped; past the chunk's end the robot stops."""

    def __init__(self, K):
        self.K = K
        self.seq = -1
        self.obs_step = None
        self.chunk = None
        self.stale = 0

    def on_reply(self, seq, obs_step, actions):
        if seq <= self.seq:
            self.stale += 1
            return False
        self.seq, self.obs_step = seq, obs_step
        self.chunk = np.asarray(actions, dtype=float).reshape(-1, 3)
        return True

    def command(self, step):
        if self.chunk is None:
            return np.zeros(3)
        if self.K == 1:
            return self.chunk[0]
        k = step - self.obs_step
        return self.chunk[k] if 0 <= k < len(self.chunk) else np.zeros(3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class Registry:
    """Versioned ONNX policies with their model cards. A model is deployable only if its card says it passed the
    release gate; the server refuses the others unless explicitly allowed (experiments)."""

    def __init__(self, model_dir, threads=1):
        import onnxruntime as ort
        self.models = {}
        for card_path in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
            with open(card_path) as f:
                card = json.load(f)
            onnx_path = os.path.join(model_dir, card["name"] + ".onnx")
            if not os.path.exists(onnx_path):
                continue
            if card.get("sha256") and card["sha256"] != sha256(onnx_path):
                raise RuntimeError(f"{card['name']}: ONNX file does not match its model card checksum")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
            self.models[card["name"]] = dict(card=card, session=sess)

    def names(self):
        return list(self.models)

    def card(self, name):
        return self.models[name]["card"]

    def default(self):
        dep = [n for n, m in self.models.items() if m["card"].get("deployable")]
        return dep[-1] if dep else None

    def run(self, name, obs_batch):
        return self.models[name]["session"].run(None, {"obs": obs_batch})[0]
