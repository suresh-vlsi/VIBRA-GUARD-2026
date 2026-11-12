"""Unsupervised health baseline: Mahalanobis detector with continual (online) learning."""
import numpy as np
from .dsp import ANOM_KEYS, vec


class HealthBaseline:
    def __init__(self, alpha=0.02):
        self.alpha, self.thr = alpha, None

    def fit(self, feats):
        X = np.array([vec(f, ANOM_KEYS) for f in feats])
        self.mu = X.mean(0)
        self.cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-3
        self.rms0 = float(np.median([f["rms"] for f in feats]))
        d = np.array([self.score(f)[0] for f in feats])
        self.thr = float(np.percentile(d, 99) * 1.5)
        return self

    def score(self, f):
        x = vec(f, ANOM_KEYS)
        diff = x - self.mu
        dp = np.maximum(diff, 0)          # one-sided: only *more* vibration is abnormal
        d = float(np.sqrt(dp @ np.linalg.solve(self.cov, dp)))
        z = diff / np.sqrt(np.diag(self.cov))
        return d, z

    def explain(self, f, top=3):
        _, z = self.score(f)
        idx = np.argsort(-np.abs(z))[:top]
        return [(ANOM_KEYS[i], round(float(z[i]), 1)) for i in idx]

    def update(self, f):
        """On-device continual learning of the healthy baseline (EWMA)."""
        x = vec(f, ANOM_KEYS)
        diff = x - self.mu
        self.mu = self.mu + self.alpha * diff
        self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(diff, diff)
        self.rms0 = (1 - self.alpha) * self.rms0 + self.alpha * f["rms"]
