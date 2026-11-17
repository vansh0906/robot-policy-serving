"""Demonstrations from the scripted expert, with optional DART noise and action-chunk labels."""
import numpy as np

from . import arm

NARROW = dict(r_range=(0.30, 0.50), z_range=(0.15, 0.45), yaw=(-1.5, 1.5))     # research lab's original data
FULL = dict(r_range=(0.20, 0.62), z_range=(0.02, 0.62), yaw=(-2.4, 2.4))       # the whole reachable workspace


def sample_targets(rng, n, r_range, z_range, yaw):
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        r, z, a = rng.uniform(*r_range, m), rng.uniform(*z_range, m), rng.uniform(*yaw, m)
        p = np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)
        out.append(p[~np.isnan(arm.ik_batch(p)[:, 0])])
    return np.concatenate(out)[:n]


def sample_starts(rng, n):
    q = arm.Q_HOME + rng.normal(0, [0.8, 0.3, 0.4], (n, 3))
    return np.clip(q, arm.Q_LO + 0.05, arm.Q_HI - 0.05)


def chunk_labels(q, qd, target, K):
    """The expert's next K commands from each state, rolled out without noise (actuator lag included)."""
    out = np.zeros((len(q), K, 3), np.float32)
    for k in range(K):
        u = arm.expert_batch(q, target)
        out[:, k] = u
        q, qd, _ = arm.step_batch(q, qd, u)
    return out.reshape(len(q), K * 3)


def demos(n_ep, K=1, noise=0.0, dist=FULL, seed=0, steps=None):
    """Expert episodes (half with a target change at 2 s, as when a user moves the target mid-motion).
    With noise > 0 (DART), the executed command is perturbed but the label stays the expert's."""
    rng = np.random.default_rng(seed)
    steps = steps or int(arm.EPISODE_S / arm.DT_CTRL)
    q, qd = sample_starts(rng, n_ep), np.zeros((n_ep, 3))
    tgt = sample_targets(rng, n_ep, **dist)
    tgt2 = sample_targets(rng, n_ep, **dist)
    switch = rng.random(n_ep) < 0.5
    X, S = [], []
    for t in range(steps):
        if t == steps // 2:
            tgt = np.where(switch[:, None], tgt2, tgt)
        X.append(arm.observation_batch(q, qd, tgt))
        S.append((q.copy(), qd.copy(), tgt.copy()))
        u = arm.expert_batch(q, tgt)
        if noise > 0:
            u = u + rng.normal(0, noise, u.shape)
        q, qd, _ = arm.step_batch(q, qd, u)
    X = np.concatenate(X).astype(np.float32)
    Q = np.concatenate([s[0] for s in S])
    QD = np.concatenate([s[1] for s in S])
    T = np.concatenate([s[2] for s in S])
    return X, chunk_labels(Q, QD, T, K)
