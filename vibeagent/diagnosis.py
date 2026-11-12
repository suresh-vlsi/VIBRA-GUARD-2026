"""Fault diagnosis: Random Forest on order features fused with physics rules."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .config import *
from .simulator import Machine
from .dsp import preprocess, features, vec, DIAG_KEYS


def _near_res(rpm, fn):
    fr = rpm / 60
    return any(abs(k * fr - fn) < 0.14 * fn for k in (1, 2, 3))


def simulate_window(fault, rng, seed):
    """One labelled training/test window from a randomised machine."""
    fn = rng.uniform(40, 100)
    if fault == "resonance":
        k = 2 if (60 * fn / 1 > 4500 or rng.random() < 0.7) else 1
        rpm = 60 * fn / k * (1 + rng.uniform(-0.05, 0.05))
        sev = 0.0
    else:
        while True:
            rpm = rng.uniform(800, 2000)
            if not _near_res(rpm, fn):
                break
        sev = rng.uniform(0.4, 1.0)
    m = Machine(rpm=rpm, fn=fn, seed=seed)
    m.inject("healthy" if fault == "resonance" else fault, sev)
    m.step(WIN)                               # settle
    s = preprocess(m.step(WIN))
    fn_est = fn * (1 + rng.normal(0, 0.01))
    return features(s, rpm, fn_est)


def make_dataset(n_per_class=120, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(FAULTS):
        for i in range(n_per_class):
            X.append(vec(simulate_window(c, rng, seed * 100000 + ci * 1000 + i), DIAG_KEYS)); y.append(ci)
    return np.array(X), np.array(y)


def rule_scores(F):
    """Physics-based evidence in [0,1] per class + human-readable reasons."""
    sat = lambda x, lo, hi: float(np.clip((x - lo) / (hi - lo), 0, 1))
    r, why = np.zeros(len(FAULTS)), {}
    ix = {c: i for i, c in enumerate(FAULTS)}
    pres = max(F["prox1"], F["prox2"], F["prox3"])
    r[ix["bearing_outer"]] = min(sat(F["env_bpfo"], 4, 10), sat(F["kurt"], 3, 6))
    r[ix["bearing_inner"]] = min(sat(F["env_bpfi"], 4, 10), sat(F["kurt"], 3, 6))
    r[ix["looseness"]] = max(sat(F["sub_a1"], .25, .8), sat(F["harm_a1"], .5, 1.5)) * sat(F["rms"], .06, .12)
    r[ix["resonance"]] = sat(pres, .4, .8) * sat(F["a2_rms"] + F["a1_rms"], .8, 1.0) * sat(F["rms"], .15, .4)
    r[ix["misalignment"]] = sat(F["a2_a1"], .6, 1.5) * sat(F["a2_rms"], .6, .9) * (1 - sat(pres, .4, .8))
    r[ix["imbalance"]] = sat(F["a1_rms"], .75, .95) * (1 - sat(F["a2_a1"], .3, .6)) * (1 - sat(pres, .4, .8))
    why["bearing_outer"] = f"envelope peak at BPFO SNR={F['env_bpfo']:.1f}, kurtosis={F['kurt']:.1f}"
    why["bearing_inner"] = f"envelope peak at BPFI SNR={F['env_bpfi']:.1f}, kurtosis={F['kurt']:.1f}"
    why["looseness"] = f"sub-harm/1x={F['sub_a1']:.2f}, harm(3-5x)/1x={F['harm_a1']:.2f}"
    why["resonance"] = f"shaft order within 8% of structural fn (prox={pres:.2f}), strong narrowband response"
    why["misalignment"] = f"2x/1x amplitude ratio={F['a2_a1']:.2f}"
    why["imbalance"] = f"1x dominates spectrum ({F['a1_rms']*100:.0f}% of RMS), 2x/1x={F['a2_a1']:.2f}"
    return r, why


class Diagnoser:
    def __init__(self, w_rules=0.35):
        self.rf = RandomForestClassifier(n_estimators=80, max_depth=9, random_state=0, n_jobs=-1)
        self.w_rules = w_rules

    def fit(self, X, y):
        self.rf.fit(X, y); return self

    def probs(self, F):
        p = self.rf.predict_proba(vec(F, DIAG_KEYS)[None])[0]
        r, why = rule_scores(F)
        if r.sum() > 0:
            r = r / r.sum()
            p = (1 - self.w_rules) * p + self.w_rules * r
        return p / p.sum(), why

    def diagnose(self, feats):
        """Average probabilities over several windows -> (label, confidence, reason)."""
        ps = [self.probs(F) for F in feats]
        p = np.mean([x[0] for x in ps], 0)
        i = int(np.argmax(p))
        return FAULTS[i], float(p[i]), ps[-1][1].get(FAULTS[i], "no fault signature")
