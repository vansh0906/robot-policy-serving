"""Closed-loop evaluation of a policy on many arms at once, with a network-delay model.

At each 20 ms step every robot sends its observation. The reply to the request sent at step s arrives at step
s + d, where d = floor(RTT / 20 ms). Modes:
  hold   one command per reply; the robot keeps applying the newest command it has (newest by sequence number)
  chunk  K commands per reply; the robot applies element (t - s) of the newest chunk, i.e. it skips the commands
         whose time has already passed; past the end of the chunk it stops (zero velocity)
  chunk_noskip  ablation: plays the chunk from its first element when it arrives, ignoring time in flight
"""
import numpy as np

from . import arm
from .data import FULL, sample_starts, sample_targets


def tasks(n, seed=123, dist=FULL):
    rng = np.random.default_rng(seed)
    return sample_starts(rng, n), sample_targets(rng, n, **dist)


def rtt_steps(rng, shape, rtt_ms, jitter_ms):
    """Round-trip time = fixed network part + exponential jitter, in whole control periods."""
    rtt = rtt_ms + (rng.exponential(jitter_ms, shape) if jitter_ms > 0 else np.zeros(shape))
    return np.floor(rtt / (arm.DT_CTRL * 1e3)).astype(int)


def run(policy, starts, targets, K=1, mode="hold", rtt_ms=0.0, jitter_ms=0.0, seed=7, record=False):
    rng = np.random.default_rng(seed)
    n = len(starts)
    T = int(arm.EPISODE_S / arm.DT_CTRL)
    q, qd = starts.copy(), np.zeros((n, 3))
    outs = np.zeros((T, n, K, 3))
    arrive = rtt_steps(rng, (T, n), rtt_ms, jitter_ms)
    best = np.full(n, -1)                          # newest request whose reply has arrived
    got = np.full(n, -1)                           # step at which that reply arrived
    hold = np.zeros(n, int)
    done_t = np.full(n, np.nan)
    hits = np.zeros(n, bool)
    traj = []
    rows = np.arange(n)
    for t in range(T):
        obs = arm.observation_batch(q, qd, targets)
        outs[t] = policy(obs).reshape(n, K, 3)
        for s in range(max(0, t - 60), t + 1):     # replies landing now (RTT up to 1.2 s)
            land = (s + arrive[s] == t) & (s > best)
            best = np.where(land, s, best)
            got = np.where(land, t, got)
        u = np.zeros((n, 3))
        have = best >= 0
        if mode == "hold":
            u[have] = outs[best[have], rows[have], 0]
        else:
            k = t - (best if mode == "chunk" else got)
            ok = have & (k < K)
            u[ok] = outs[best[ok], rows[ok], k[ok]]
        q, qd, hit = arm.step_batch(q, qd, u)
        hits |= hit
        err = np.linalg.norm(arm.fk(q) - targets, axis=1)
        hold = np.where(err < arm.TOL, hold + 1, 0)
        newly = np.isnan(done_t) & (hold * arm.DT_CTRL >= arm.HOLD_S - 1e-9)
        done_t[newly] = (t + 1) * arm.DT_CTRL
        if record:
            traj.append(q.copy())
    res = dict(success=float(np.mean(~np.isnan(done_t))),
               t_median=float(np.nanmedian(done_t)) if np.any(~np.isnan(done_t)) else None,
               final_err_mm_median=float(np.median(err) * 1e3),
               limit_hit_rate=float(hits.mean()),
               ok=~np.isnan(done_t))
    if record:
        res["traj"] = np.array(traj)
    return res
