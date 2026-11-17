"""Policy network: a small MLP trained by behaviour cloning, in plain NumPy (the "research" side).

Output: a chunk of K future joint-velocity commands (K x 3, normalised to [-1, 1] by a tanh output layer).
"""
import json
import os

import numpy as np


class MLP:
    def __init__(self, sizes, rng=None):
        rng = rng or np.random.default_rng(0)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mu = np.zeros(sizes[0], np.float32)
        self.sd = np.ones(sizes[0], np.float32)

    def forward(self, x):
        h = (np.asarray(x, np.float32) - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = np.tanh(h @ W + b)                   # tanh hidden layers, tanh output (bounded commands)
        return h

    def fit(self, X, Y, epochs=60, batch=512, lr=2e-3, seed=0, log=None):
        rng = np.random.default_rng(seed)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))          # cosine decay
            tot = 0.0
            for s in range(0, n, batch):
                j = idx[s:s + batch]
                hs = [Xn[j]]
                for W, b in zip(self.W, self.b):
                    hs.append(np.tanh(hs[-1] @ W + b))
                err = hs[-1] - Y[j]
                tot += float((err ** 2).sum())
                g = 2 * err / len(j)
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for i in reversed(range(len(self.W))):
                    g = g * (1 - hs[i + 1] ** 2)
                    gW[i] = hs[i].T @ g
                    gb[i] = g.sum(0)
                    g = g @ self.W[i].T
                t += 1
                for k, (p, gr) in enumerate(zip(params, gW + gb)):
                    m[k] = 0.9 * m[k] + 0.1 * gr
                    v[k] = 0.999 * v[k] + 0.001 * gr * gr
                    p -= lr_ep * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if log:
                log(ep, tot / n / Y.shape[1])
        return self

    def save(self, path, meta):
        np.savez(path, mu=self.mu, sd=self.sd, **{f"W{i}": W for i, W in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        W = [z[f"W{i}"] for i in range(n)]
        obj = cls([W[0].shape[0]] + [w.shape[1] for w in W])
        obj.W, obj.b = W, [z[f"b{i}"] for i in range(n)]
        obj.mu, obj.sd = z["mu"], z["sd"]
        return obj
