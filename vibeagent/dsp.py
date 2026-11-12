"""Pre-processing and feature extraction (all rpm-aware, order based)."""
import numpy as np
from scipy import signal, stats
from .config import *

_SOS_HP = signal.butter(4, 5, "highpass", fs=FS, output="sos")
_SOS_BP = signal.butter(4, [450, 900], "bandpass", fs=FS, output="sos")
_WIN = signal.windows.hann(WIN)
_FREQ = np.fft.rfftfreq(WIN, 1 / FS)


def preprocess(x):
    """(n,3) raw IMU -> 1-D principal-axis, gravity-free vibration signal (g)."""
    x = x - x.mean(0)
    x = signal.sosfiltfilt(_SOS_HP, x, axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[0]


def amp_spectrum(s):
    return 2 * np.abs(np.fft.rfft(s * _WIN)) / (WIN * _WIN.mean())


def _peak(A, f0, tol=2.5):
    m = (_FREQ > f0 - tol) & (_FREQ < f0 + tol)
    return A[m].max() if m.any() else 0.0


def features(s, rpm, fn_est):
    fr = max(rpm / 60.0, 1e-3)
    A = amp_spectrum(s)
    F = {"rms": float(np.sqrt(np.mean(s ** 2))), "kurt": float(stats.kurtosis(s, fisher=False))}
    F["crest"] = float(np.abs(s).max() / (F["rms"] + 1e-9))
    for name, k in [("a05", .5), ("a1", 1), ("a15", 1.5), ("a2", 2), ("a3", 3), ("a4", 4), ("a5", 5)]:
        F[name] = float(_peak(A, k * fr))
    e = F["a1"] + 1e-6
    F["a2_a1"] = F["a2"] / e
    F["harm_a1"] = (F["a3"] + F["a4"] + F["a5"]) / e
    F["sub_a1"] = (F["a05"] + F["a15"]) / e
    F["a1_rms"] = F["a1"] / (F["rms"] + 1e-9)
    F["a2_rms"] = F["a2"] / (F["rms"] + 1e-9)
    # bearing envelope analysis
    env = np.abs(signal.hilbert(signal.sosfiltfilt(_SOS_BP, s)))
    E = np.abs(np.fft.rfft((env - env.mean()) * _WIN)) / WIN
    band = (_FREQ > 5) & (_FREQ < 300)
    med = np.median(E[band]) + 1e-9
    F["env_bpfo"] = float(_peak(E, BPFO * fr, 3.0) / med)
    F["env_bpfi"] = float(_peak(E, BPFI * fr, 3.0) / med)
    # proximity of shaft orders to the structural resonance
    for k in (1, 2, 3):
        F[f"prox{k}"] = float(np.exp(-((k * fr - fn_est) / (0.08 * fn_est)) ** 2))
    return F


DIAG_KEYS = ["rms", "kurt", "crest", "a1_rms", "a2_rms", "a2_a1", "harm_a1", "sub_a1",
             "env_bpfo", "env_bpfi", "prox1", "prox2", "prox3"]
ANOM_KEYS = ["rms", "a05", "a1", "a15", "a2", "a3", "a4", "a5", "kurt", "crest", "env_bpfo", "env_bpfi"]
LOG_KEYS = {"rms", "a05", "a1", "a15", "a2", "a3", "a4", "a5"}


def vec(F, keys):
    """Feature vector for ``keys``; anomaly features are log-scaled (compared by value, not identity)."""
    use_log = list(keys) == ANOM_KEYS
    return np.array([np.log(F[k] + 1e-4) if (use_log and k in LOG_KEYS) else F[k] for k in keys])
