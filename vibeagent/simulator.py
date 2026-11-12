"""Physics-based machine + 3-axis IMU simulator.

Structure = low mode (fn, driven by rotating forces and the actuator) plus a high
mode (650 Hz, rung by bearing-defect impacts).  Acceleration units are g.
"""
import numpy as np
from scipy import signal
from .config import *


def _accel_tf(fn, zeta, fs=FS):
    """Digital accel/force transfer function s^2/(s^2+2*z*w*s+w^2) (prewarped bilinear)."""
    wn = 2 * np.pi * fn
    wp = 2 * fs * np.tan(wn / (2 * fs))
    return signal.bilinear([1, 0, 0], [1, 2 * zeta * wp, wp ** 2], fs)


def H_mag(f, fn, zeta=ZETA):
    """Analytic |accel/force| used by the speed-planning logic."""
    r = np.asarray(f, float) / fn
    return r ** 2 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)


class Machine:
    GAINS = np.array([1.0, 0.55, 0.3])

    def __init__(self, rpm=RPM_NOM, fn=FN_HEALTHY, seed=0):
        self.rng = np.random.default_rng(seed)
        self.rpm, self.fn = float(rpm), float(fn)
        self.fault, self.sev = "healthy", 0.0
        self.theta = self.rng.uniform(0, 2 * np.pi)
        self.p = self.rng.uniform(0, 2 * np.pi, 8)
        self.grav = np.array([0.02, 0.03, 1.0]) + self.rng.normal(0, 0.01, 3)
        self.stopped = False
        self.time = 0.0
        self._next = {"o": 50.0, "i": 80.0}
        self._build()

    # ---- configuration -------------------------------------------------
    def _build(self):
        self.b_lo, self.a_lo = _accel_tf(self.fn, ZETA)
        self.b_hi, self.a_hi = _accel_tf(650.0, 0.05)
        self.zi_lo, self.zi_hi = np.zeros(2), np.zeros(2)

    def set_fn(self, fn):
        self.fn = float(fn); self._build()

    def set_rpm(self, rpm):
        self.rpm = float(np.clip(rpm, 0, RPM_MAX * 3))

    def inject(self, fault, sev=0.8):
        self.fault, self.sev = fault, sev

    def peek_phase(self, n):
        """Tacho reference: shaft phase for the next n samples (rad)."""
        return self.theta + 2 * np.pi * (self.rpm / 60) * np.arange(1, n + 1) / FS

    # ---- dynamics ----------------------------------------------------------
    def _forces(self, n):
        fr, sc = self.rpm / 60, (self.rpm / RPM_NOM) ** 2
        ph = self.peek_phase(n)
        u, m2, sev, p = 0.3, 0.05, self.sev, self.p
        f = np.zeros(n)
        if self.fault == "imbalance":
            u += 1.5 * sev
        elif self.fault == "misalignment":
            m2 += 0.8 * sev; u += 0.3 * sev
        f += sc * (u * np.sin(ph + p[0]) + m2 * np.sin(2 * ph + p[1]) + 0.01 * np.sin(3 * ph + p[2]))
        if self.fault == "looseness":
            a = 0.5 * sev
            for k, w in [(0.5, 0.6), (1.5, 0.4), (3, 0.5), (4, 0.4), (5, 0.3)]:
                f += sc * a * w * np.sin(k * ph + p[3] + k)
            f += sc * 0.4 * sev * np.sin(ph + p[0])
        f += self.rng.normal(0, 0.02, n)            # broadband process excitation
        imp = np.zeros(n)
        if self.fault in ("bearing_outer", "bearing_inner") and fr > 0:
            key, order = ("o", BPFO) if self.fault == "bearing_outer" else ("i", BPFI)
            T = FS / (order * fr)
            while self._next[key] < n:
                i = int(self._next[key])
                amp = 0.35 * sev * (sc ** 0.5) * self.rng.uniform(0.7, 1.3)
                if key == "i":
                    amp *= 1 + 0.8 * np.cos(ph[i])
                imp[i] += amp
                self._next[key] += T * (1 + self.rng.normal(0, 0.01))
            self._next[key] -= n
        return f, imp

    def step(self, n, act=None):
        """Advance n samples; act = actuator force (n,) -> returns IMU data (n,3) in g."""
        if self.stopped:
            self.rpm = 0.0
        f, imp = self._forces(n) if self.rpm > 0 else (np.zeros(n), np.zeros(n))
        if act is not None:
            f = f + np.clip(act, -ACT_LIMIT, ACT_LIMIT)
        a_lo, self.zi_lo = signal.lfilter(self.b_lo, self.a_lo, f, zi=self.zi_lo)
        a_hi, self.zi_hi = signal.lfilter(self.b_hi, self.a_hi, imp, zi=self.zi_hi)
        a = a_lo + a_hi
        self.theta += 2 * np.pi * (self.rpm / 60) * n / FS
        self.time += n / FS
        self.grav += self.rng.normal(0, 1e-4, 3)       # slow drift
        out = self.GAINS[None, :] * a[:, None] + self.rng.normal(0, 0.01, (n, 3)) + self.grav
        return out

    def probe(self, n=8192, sigma=1.0):
        """Actuator white-noise probe for system identification."""
        u = self.rng.normal(0, sigma, n)
        y = self.step(n, u)[:, 0]
        return np.clip(u, -ACT_LIMIT, ACT_LIMIT), y - y.mean()
