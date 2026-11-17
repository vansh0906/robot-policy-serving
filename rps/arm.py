"""3-DOF arm (base yaw, shoulder pitch, elbow pitch) with velocity-controlled joints.

The robot takes joint-velocity commands at 50 Hz, like most commercial arms in velocity mode. Each joint follows the
command through a first-order lag with velocity and acceleration limits, integrated at 1 kHz. The task is reaching a
3-D target with the tool tip and holding it.
"""
from dataclasses import dataclass, field

import numpy as np

H0, L1, L2 = 0.30, 0.35, 0.33                        # base height, upper arm, forearm + tool (m)
Q_LO = np.array([-2.6, -0.2, -2.7])                  # joint limits (rad)
Q_HI = np.array([2.6, 1.9, -0.15])
V_MAX = 1.5                                          # joint velocity limit (rad/s)
A_MAX = 10.0                                         # joint acceleration limit (rad/s^2)
TAU = 0.06                                           # actuator lag (s)
DT_CTRL = 0.02                                       # control period (s), 50 Hz
N_SUB = 20                                           # physics substeps per control step (1 kHz)
Q_HOME = np.array([0.0, 0.9, -1.6])
OBS_DIM = 18
TOL = 0.015                                          # success: tip within 1.5 cm ...
HOLD_S = 0.3                                         # ... held for 0.3 s
EPISODE_S = 4.0


def fk(q):
    """Tool-tip position for joint angles q (..., 3)."""
    q = np.asarray(q, dtype=float)
    r = L1 * np.cos(q[..., 1]) + L2 * np.cos(q[..., 1] + q[..., 2])
    z = H0 + L1 * np.sin(q[..., 1]) + L2 * np.sin(q[..., 1] + q[..., 2])
    return np.stack([r * np.cos(q[..., 0]), r * np.sin(q[..., 0]), z], axis=-1)


def link_points(q):
    """Base, shoulder, elbow and tip positions (for the viewer and tests)."""
    q0, q1, q2 = q
    c, s = np.cos(q0), np.sin(q0)
    sh = np.array([0.0, 0.0, H0])
    re, ze = L1 * np.cos(q1), H0 + L1 * np.sin(q1)
    el = np.array([re * c, re * s, ze])
    return np.array([[0.0, 0.0, 0.0], sh, el, fk(q)])


def jacobian(q):
    q0, q1, q2 = q
    r = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    dr1 = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    dr2 = -L2 * np.sin(q1 + q2)
    dz1 = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    dz2 = L2 * np.cos(q1 + q2)
    c, s = np.cos(q0), np.sin(q0)
    return np.array([[-r * s, dr1 * c, dr2 * c],
                     [r * c, dr1 * s, dr2 * s],
                     [0.0, dz1, dz2]])


def ik(p, elbow_up=True):
    """Analytic IK for a tip position; returns None when out of reach or joint limits."""
    x, y, z = p
    r, h = np.hypot(x, y), z - H0
    d2 = r * r + h * h
    c2 = (d2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None
    q2 = -np.arccos(c2) if elbow_up else np.arccos(c2)
    q1 = np.arctan2(h, r) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q = np.array([np.arctan2(y, x), q1, q2])
    if np.any(q < Q_LO) or np.any(q > Q_HI):
        return None
    return q


def sample_target(rng, r_range=(0.20, 0.62), z_range=(0.02, 0.62), yaw=(-2.4, 2.4)):
    """A reachable target (the IK solution must respect joint limits)."""
    while True:
        r, z, a = rng.uniform(*r_range), rng.uniform(*z_range), rng.uniform(*yaw)
        p = np.array([r * np.cos(a), r * np.sin(a), z])
        if ik(p) is not None:
            return p


def sample_start(rng):
    """A random start configuration around home."""
    return np.clip(Q_HOME + rng.normal(0, [0.8, 0.3, 0.4]), Q_LO + 0.05, Q_HI - 0.05)


def observation(q, qd, target):
    """The 18-number observation the robot sends to the server (radians, rad/s, metres)."""
    ee = fk(q)
    return np.concatenate([np.sin(q), np.cos(q), qd / V_MAX, target, ee, (target - ee) * 5.0])


def expert(q, target, gain=3.0, v_frac=0.8):
    """Scripted expert: analytic IK goal, then a synchronised joint-space move (a straight line between two in-limit
    configurations never leaves the joint-limit box). Returns normalised velocity commands in [-1, 1]."""
    q_goal = ik(target)
    if q_goal is None:
        return np.zeros(3)
    qd = gain * (q_goal - q)
    m = np.max(np.abs(qd)) / (v_frac * V_MAX)
    if m > 1:
        qd /= m
    return qd / V_MAX


@dataclass
class Arm:
    q: np.ndarray = field(default_factory=lambda: Q_HOME.copy())
    qd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    limit_hits: int = 0

    def step(self, u_norm):
        """Apply a normalised velocity command for one 20 ms control period."""
        u = np.clip(np.asarray(u_norm, dtype=float), -1, 1) * V_MAX
        h = DT_CTRL / N_SUB
        for _ in range(N_SUB):
            acc = np.clip((u - self.qd) / TAU, -A_MAX, A_MAX)
            self.qd = np.clip(self.qd + acc * h, -V_MAX, V_MAX)
            q = self.q + self.qd * h
            hit = (q < Q_LO) | (q > Q_HI)
            if hit.any():
                self.limit_hits += 1
                q = np.clip(q, Q_LO, Q_HI)
                self.qd[hit] = 0.0
            self.q = q
        return self.q


# ---- batched versions (data generation and evaluation run many arms at once) ----

def ik_batch(p):
    """Vectorised elbow-up IK; rows that are unreachable come back as NaN."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r, h = np.hypot(x, y), z - H0
    c2 = (r * r + h * h - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    bad = np.abs(c2) > 1
    q2 = -np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(h, r) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q = np.stack([np.arctan2(y, x), q1, q2], axis=1)
    bad |= np.any((q < Q_LO) | (q > Q_HI), axis=1)
    q[bad] = np.nan
    return q


def expert_batch(q, target, gain=3.0, v_frac=0.8):
    qg = ik_batch(target)
    qd = gain * (qg - q)
    m = np.max(np.abs(qd), axis=1, keepdims=True) / (v_frac * V_MAX)
    qd = np.where(m > 1, qd / np.maximum(m, 1e-12), qd)
    return np.nan_to_num(qd / V_MAX)


def step_batch(q, qd, u_norm):
    """One 20 ms control period for N arms; returns new (q, qd, limit_hit_mask)."""
    u = np.clip(u_norm, -1, 1) * V_MAX
    h = DT_CTRL / N_SUB
    hit_any = np.zeros(len(q), dtype=bool)
    for _ in range(N_SUB):
        acc = np.clip((u - qd) / TAU, -A_MAX, A_MAX)
        qd = np.clip(qd + acc * h, -V_MAX, V_MAX)
        qn = q + qd * h
        hit = (qn < Q_LO) | (qn > Q_HI)
        hit_any |= hit.any(axis=1)
        q = np.clip(qn, Q_LO, Q_HI)
        qd = np.where(hit, 0.0, qd)
    return q, qd, hit_any


def observation_batch(q, qd, target):
    ee = fk(q)
    return np.concatenate([np.sin(q), np.cos(q), qd / V_MAX, target, ee, (target - ee) * 5.0], axis=1)
