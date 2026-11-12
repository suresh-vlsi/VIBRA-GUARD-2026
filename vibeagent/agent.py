"""Agentic decision layer: sense -> detect -> probe -> diagnose -> plan -> act -> verify -> learn."""
import json
from collections import deque
import numpy as np
from .config import *
from .simulator import H_mag
from .dsp import preprocess, features

COST = {"ACTIVE_REJECT": 0.0, "SHIFT_SPEED": 0.05, "OPTIMIZE_SPEED": 0.10, "DERATE": 0.25, "SAFE_STOP": 1.0}
# prior expected RMS reduction of each action per fault (refined online from outcomes)
PRIOR = {
    "imbalance":     {"ACTIVE_REJECT": .85, "DERATE": .55},
    "misalignment":  {"ACTIVE_REJECT": .80, "DERATE": .55},
    "resonance":     {"SHIFT_SPEED": .85, "ACTIVE_REJECT": .70, "DERATE": .50},
    # These faults use measured speed trials instead of assuming a fixed
    # 30% derate is beneficial. Candidate RPMs are accepted only after
    # measured RMS improvement and the normal post-action verification.
    "looseness":     {"OPTIMIZE_SPEED": .75},
    "bearing_outer": {"OPTIMIZE_SPEED": .70},
    "bearing_inner": {"DERATE": .50},
}
WAIT = {"ACTIVE_REJECT": 4, "SHIFT_SPEED": 2, "OPTIMIZE_SPEED": 2, "DERATE": 2}
SUCCESS_REDUCTION = 0.35        # >= ~3.7 dB RMS drop counts as success


class Agent:
    def __init__(self, machine, baseline, diagnoser, rejector, log_path=None, verbose=True):
        self.m, self.base, self.diag, self.rej = machine, baseline, diagnoser, rejector
        # Process setpoint = the speed the machine is started at (not a hard-coded 1200 rpm).
        self.setpoint = float(machine.rpm)
        self.rms_before = None            # pre-action RMS reference (set when an incident opens)
        self.wait = 0                     # windows left in VERIFY
        self.verbose, self.log_path = verbose, log_path
        self.mem = {}                       # learned (fault, action) -> expected reduction
        self.state = "MONITOR"
        self.win, self.fn_est = 0, FN_HEALTHY
        self.over = deque(maxlen=3)
        self.buf = deque(maxlen=3)
        self.rows, self.events, self.sigs = [], [], []
        self.plan, self.cur, self.failed = [], None, set()
        self.fault = None
        self.action_applied = False
        self.minor_rms = None
        self.quiet = 0
        self.mit_rms, self.bad = None, 0

        # Action verification uses several post-action windows rather than
        # trusting one sample. Recovery is a separate, stricter condition.
        self.verify_rms = []
        self.verify_scores = []
        self.recovery_elapsed = 0
        self.recovery_quiet_required = 6
        self.recovery_timeout = 20
        self.last_outcome = "NOT_RUN"
        if log_path:
            open(log_path, "w").close()

    @property
    def recovery_rms_limit(self):
        """Absolute RMS bound for 'recovered', tied to the *current* healthy baseline."""
        return max(1.5 * self.base.rms0, self.base.rms0 + 0.02)

    # ---- utilities -------------------------------------------------------
    def calibrate(self):
        self.fn_est = self.rej.identify(self.m)
        # Allow probe-induced structural vibration to settle before monitoring.
        self.m.step(FS)
        self._ev("CALIBRATE", f"identified structural fn = {self.fn_est:.1f} Hz")

    def _ev(self, kind, msg):
        e = {"win": self.win, "t": round(self.m.time, 2), "type": kind, "msg": msg}
        self.events.append(e)
        if self.verbose:
            print(f"  [w{self.win:03d} t={self.m.time:5.1f}s] {kind:<11} {msg}")
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(e) + "\n")

    def _score(self, a, fault):
        pri = self.mem.get((fault, a), PRIOR[fault].get(a, 0.0))
        return pri - COST[a]

    # ---- main loop step ---------------------------------------------------
    def tick(self, truth=None):
        m = self.m
        act = self.rej.output(m, WIN) if self.rej.enabled else None
        raw = m.step(WIN, act)
        if self.rej.enabled:
            self.rej.update(raw[:, 0])
        self.win += 1
        s = preprocess(raw)
        F = features(s, m.rpm, self.fn_est)
        d, _ = self.base.score(F)
        over = d > self.base.thr
        self.over.append(over)
        self.buf.append((s, m.rpm, over))
        self.sigs.append(s)
        row = dict(win=self.win, t=m.time, rpm=m.rpm, rms=F["rms"], score=d / self.base.thr,
                   state=self.state, truth=truth, action=self.cur)
        self.rows.append(row)

        if self.state == "STOPPED":
            return row
        # A hard safety trip overrides every operating state, including VERIFY.
        if F["rms"] > RMS_TRIP:
            self._ev("TRIP", f"RMS {F['rms']:.2f} g above safety limit")
            self._safe_stop("hard limit exceeded"); return row

        if self.state == "MONITOR":
            trigger = sum(self.over) >= 2 and over
            if self.minor_rms and F["rms"] < 1.5 * self.minor_rms:
                trigger = False
            if trigger:
                self._open_incident(F)
            elif not over and d < 0.7 * self.base.thr and not self.action_applied:
                self.base.update(F)          # continual learning of the healthy baseline
        elif self.state == "VERIFY":
            self.verify_rms.append(float(F["rms"]))
            self.verify_scores.append(float(d))
            self.wait -= 1
            if self.wait <= 0:
                self._verify(F, d)
        elif self.state == "MITIGATED":
            self.recovery_elapsed += 1

            # A quiet window must satisfy both the anomaly detector and an
            # absolute RMS bound tied to the learned healthy baseline.
            recovered_window = (
                (not over)
                and (d <= self.base.thr)
                and (F["rms"] <= self.recovery_rms_limit)
            )
            self.quiet = self.quiet + 1 if recovered_window else 0

            self.bad = self.bad + 1 if (
                F["rms"] > max(1.5 * self.mit_rms, self.recovery_rms_limit)
            ) else 0

            if self.bad >= 2:
                self._ev("DEGRADED", "vibration rose again after mitigation -> escalating")
                self.bad = 0
                if self.cur is not None:
                    self.failed.add(self.cur)
                if self.cur == "ACTIVE_REJECT":
                    self.rej.disable()      # do not leave a failed controller driving the actuator
                self._next_action(F["rms"])
            elif self.quiet >= self.recovery_quiet_required:
                self._ev(
                    "RECOVERED",
                    f"recovery verified: {self.quiet} consecutive healthy windows; "
                    f"RMS <= {self.recovery_rms_limit:.3f} g"
                )
                self.last_outcome = "RECOVERED"
                self.state = "MONITOR"
                self.minor_rms = None
                self.fault = None
                self.failed.clear()
                self.action_applied = False
            elif self.recovery_elapsed >= self.recovery_timeout:
                self._ev(
                    "RECOVERY_PENDING",
                    f"mitigation remains active, but recovery was not verified "
                    f"within {self.recovery_timeout} windows"
                )
                # Stay MITIGATED: do not falsely report recovery and do not
                # silently relearn this operating point as healthy.
                self.recovery_elapsed = 0
        return row

    # ---- incident handling ------------------------------------------------
    def _resonance_sweep(self, current_features):
        """Small, bounded RPM perturbation to test for a fixed structural mode.

        This is a diagnostic experiment, not a certified safety procedure.
        It compares order-response amplitudes at nearby speeds after normalizing
        for the simulator's approximate speed-squared force scaling. It never
        reads Machine.fn. Any measured hard-limit exceedance causes SAFE_STOP.
        """
        rpm0 = float(self.m.rpm)
        if rpm0 <= 0 or current_features.get("a2_a1", 0.0) < 2.0:
            return None

        scale0 = max((rpm0 / RPM_NOM) ** 2, 1e-6)
        center_a2 = float(current_features["a2"]) / scale0
        center_rms = float(current_features["rms"]) / scale0
        observations = []

        self._ev("PROBE", "testing 2x response at nearby safe speed points")
        try:
            for factor in (0.92, 1.08):
                test_rpm = float(np.clip(rpm0 * factor, RPM_MIN, RPM_MAX))
                if abs(test_rpm - rpm0) < 20:
                    continue
                self.m.set_rpm(test_rpm)
                settle_raw = self.m.step(WIN)
                settle_rms = float(np.sqrt(np.mean(preprocess(settle_raw) ** 2)))
                if settle_rms > RMS_TRIP:
                    self._ev(
                        "TRIP",
                        f"RMS {settle_rms:.2f} g during diagnostic speed-sweep settling",
                    )
                    self._safe_stop("hard limit exceeded during diagnostic sweep")
                    return None

                raw = self.m.step(WIN)
                measured = features(preprocess(raw), self.m.rpm, self.fn_est)
                if measured["rms"] > RMS_TRIP:
                    self._ev(
                        "TRIP",
                        f"RMS {measured['rms']:.2f} g during diagnostic speed sweep",
                    )
                    self._safe_stop("hard limit exceeded during diagnostic sweep")
                    return None

                scale = max((test_rpm / RPM_NOM) ** 2, 1e-6)
                observations.append({
                    "rpm": test_rpm,
                    "a2": float(measured["a2"]) / scale,
                    "rms": float(measured["rms"]) / scale,
                })
        finally:
            if self.state != "STOPPED":
                self.m.set_rpm(rpm0)

        if len(observations) < 2:
            self._ev("PROBE", "insufficient safe speed points; no resonance estimate")
            return None

        neighbor_a2 = float(np.median([x["a2"] for x in observations]))
        neighbor_rms = float(np.median([x["rms"] for x in observations]))
        a2_peak_ratio = center_a2 / max(neighbor_a2, 1e-9)
        rms_peak_ratio = center_rms / max(neighbor_rms, 1e-9)

        # A narrow response peak at the current 2x forcing frequency is
        # evidence of resonance; thresholds require later validation.
        detected = a2_peak_ratio >= 1.35 and rms_peak_ratio >= 1.20
        if detected:
            observed_mode = 2.0 * rpm0 / 60.0
            self.fn_est = float(observed_mode)
            self._ev(
                "MODEL",
                f"resonance candidate from speed sweep: estimated mode "
                f"{self.fn_est:.1f} Hz; normalized 2x peak={a2_peak_ratio:.2f}x, "
                f"RMS peak={rms_peak_ratio:.2f}x",
            )
            return {
                "detected": True,
                "mode_hz": self.fn_est,
                "a2_peak_ratio": a2_peak_ratio,
                "rms_peak_ratio": rms_peak_ratio,
            }

        self._ev(
            "PROBE",
            f"no clear fixed-mode peak: normalized 2x ratio={a2_peak_ratio:.2f}x, "
            f"RMS ratio={rms_peak_ratio:.2f}x",
        )
        return {
            "detected": False,
            "a2_peak_ratio": a2_peak_ratio,
            "rms_peak_ratio": rms_peak_ratio,
        }

    def _open_incident(self, F):
        self._ev("ANOMALY", f"score {self.rows[-1]['score']:.1f}x threshold; top deviations "
                 f"{self.base.explain(F)}")
        # Reuse the secondary-path estimate obtained during initial calibration.
        # Do not inject a new actuator probe while the machine is already faulted:
        # Machine.probe() advances the plant and adds excitation to the measured
        # vibration. If the plant dynamics have changed, handle that explicitly
        # with a safe re-identification procedure rather than probing blindly.
        self._ev(
            "MODEL",
            f"reusing calibrated secondary path; structural fn estimate = "
            f"{self.fn_est:.1f} Hz"
        )
        # If the 2x component is unusually strong, test whether its response
        # peaks sharply at the current speed. This supplies measured evidence
        # without accessing the simulator's hidden structural-frequency field.
        sweep = self._resonance_sweep(F)
        if self.state == "STOPPED":
            return

        wins = [(s, r) for s, r, o in self.buf if o] or [(self.buf[-1][0], self.buf[-1][1])]
        feats = [features(s, r, self.fn_est) for s, r in wins]
        fault, conf, why = self.diag.diagnose(feats)
        if sweep and sweep.get("detected"):
            fault = "resonance"
            conf = max(conf, 0.75)
            why = (
                f"measured speed-sweep peak near {sweep['mode_hz']:.1f} Hz; "
                f"normalized 2x response peak={sweep['a2_peak_ratio']:.2f}x"
            )
        self.fault = fault
        rms = float(np.mean([f["rms"] for f in feats]))
        ratio = rms / self.base.rms0
        sev = "minor" if ratio < 2.0 else ("moderate" if ratio < 6 else "severe")
        self._ev("DIAGNOSE", f"{fault} (conf {conf:.2f}, severity {sev}, RMS {rms:.3f} g = {ratio:.1f}x baseline) | {why}")
        if fault == "healthy":
            self._ev("DECISION", "false alarm -> keep monitoring, absorb window into baseline")
            self.base.update(feats[-1]); self.over.clear(); return
        if sev == "minor":
            self._ev("DECISION", "minor deviation -> log and keep monitoring (no intervention)")
            self.minor_rms = rms; self.over.clear(); return
        self.rms_before, self.failed, self.cur = rms, set(), None
        self.last_outcome = "FAULT_ACTIVE"
        self._next_action(rms)

    def _next_action(self, rms_ref):
        fault = self.fault
        cands = [a for a in PRIOR[fault] if a not in self.failed]
        cands = [a for a in cands if self._score(a, fault) > 0]
        cands.sort(key=lambda a: -self._score(a, fault))
        if not cands:
            self._safe_stop("no remaining mitigation options"); return
        self.rms_before = rms_ref
        a = cands[0]
        self._execute(a)

    def _optimize_speed_from_measurements(self):
        """Test bounded RPM candidates and select the lowest measured RMS.

        Each candidate gets one settling window and two measurement windows.
        The simulator's hidden structural frequency is never read. The
        candidate list is deliberately conservative and bounded by RPM_MIN.
        """
        original_rpm = float(self.m.rpm)
        # Stay inside the healthy-baseline training envelope (900–1500 rpm
        # in this project); do not "solve" a fault by extrapolating to an
        # uncalibrated low-speed operating region.
        floor = max(float(RPM_MIN), 0.75 * float(RPM_NOM))
        candidates = sorted({
            float(np.clip(floor, RPM_MIN, RPM_MAX)),
            float(np.clip(floor + 30.0, RPM_MIN, RPM_MAX)),
            float(np.clip(floor + 60.0, RPM_MIN, RPM_MAX)),
        })
        measurements = []

        self._ev(
            "PROBE",
            "measuring vibration at bounded speed candidates: "
            + ", ".join(f"{rpm:.0f}" for rpm in candidates) + " rpm",
        )

        for rpm in candidates:
            self.m.set_rpm(rpm)

            # Allow one window for the speed transition to settle.
            settle = features(preprocess(self.m.step(WIN)), self.m.rpm, self.fn_est)
            if settle["rms"] > RMS_TRIP:
                self._ev("TRIP", f"RMS {settle['rms']:.2f} g during speed-trial settling")
                self._safe_stop("hard limit exceeded during adaptive speed trial")
                return None

            rms_samples = []
            for _ in range(2):
                measured = features(preprocess(self.m.step(WIN)), self.m.rpm, self.fn_est)
                if measured["rms"] > RMS_TRIP:
                    self._ev("TRIP", f"RMS {measured['rms']:.2f} g during adaptive speed trial")
                    self._safe_stop("hard limit exceeded during adaptive speed trial")
                    return None
                rms_samples.append(float(measured["rms"]))

            candidate_rms = float(np.median(rms_samples))
            measurements.append((candidate_rms, rpm))
            self._ev(
                "PROBE",
                f"speed {rpm:.0f} rpm -> measured median RMS {candidate_rms:.3f} g",
            )

        best_rms, best_rpm = min(measurements, key=lambda item: item[0])
        required = 0.15  # require at least 15% measured improvement before changing speed
        improvement = 1.0 - best_rms / max(float(self.rms_before), 1e-9)

        if improvement < required:
            self.m.set_rpm(original_rpm)
            self._ev(
                "PROBE",
                f"no candidate achieved required {required*100:.0f}% RMS improvement; "
                f"best={best_rms:.3f} g at {best_rpm:.0f} rpm",
            )
            return None

        self.m.set_rpm(best_rpm)
        self._ev(
            "MODEL",
            f"selected measured operating point {best_rpm:.0f} rpm; "
            f"trial RMS={best_rms:.3f} g ({improvement*100:.1f}% below incident reference)",
        )
        return best_rpm

    def _execute(self, a):
        m, fault = self.m, self.fault
        self.cur, self.action_applied = a, True
        if a == "ACTIVE_REJECT":
            harm = (1,) if fault == "imbalance" else (1, 2)
            self.rej.enable(harm)
            self._ev(
                "ACTION",
                f"ACTIVE_REJECT: FxLMS cancelling orders {harm} "
                f"(using calibrated secondary path)"
            )
        elif a == "SHIFT_SPEED":
            new = self._plan_speed()
            if new is None or abs(new - m.rpm) < 10:
                self._ev("ACTION", "SHIFT_SPEED: no better speed available"); self.failed.add(a)
                self._next_action(self.rms_before); return
            self._ev("ACTION", f"SHIFT_SPEED: {m.rpm:.0f} -> {new:.0f} rpm (steering orders away from fn={self.fn_est:.0f} Hz)")
            m.set_rpm(new); self.rej.reset()
        elif a == "OPTIMIZE_SPEED":
            selected = self._optimize_speed_from_measurements()
            if selected is None:
                if self.state == "STOPPED":
                    return
                self.failed.add(a)
                self._next_action(self.rms_before)
                return
            self._ev(
                "ACTION",
                f"OPTIMIZE_SPEED: {m.rpm:.0f} rpm selected from measured candidates",
            )
            self.rej.reset()
        elif a == "DERATE":
            new = max(RPM_MIN, m.rpm * 0.7)
            self._ev("ACTION", f"DERATE: {m.rpm:.0f} -> {new:.0f} rpm")
            m.set_rpm(new); self.rej.reset()
        self._ev("ALERT", f"maintenance notified: suspected {fault}; action={a}")
        self.verify_rms = []
        self.verify_scores = []
        self.state, self.wait = "VERIFY", WAIT[a]

    def _plan_speed(self):
        """Pick the speed closest to the setpoint whose predicted vibration is acceptable."""
        grid = np.arange(RPM_MIN, RPM_MAX + 1, 25)
        pred = []
        for rpm in grid:
            fr, sc = rpm / 60, (rpm / RPM_NOM) ** 2
            pred.append(np.sqrt((0.3 * H_mag(fr, self.fn_est)) ** 2 + (0.05 * H_mag(2 * fr, self.fn_est)) ** 2
                                + (0.01 * H_mag(3 * fr, self.fn_est)) ** 2) * sc)
        pred = np.array(pred)
        ok = np.where(pred <= max(1.5 * pred.min(), 0.05))[0]
        return float(grid[ok[np.argmin(np.abs(grid[ok] - self.setpoint))]])

    def _verify(self, F, d):
        """
        Verify action efficacy using a robust summary of multiple windows.

        MITIGATED means the action materially reduced vibration.
        RECOVERED is emitted only later, after consecutive healthy windows
        satisfy both the anomaly threshold and the baseline-relative RMS limit.
        """
        samples = self.verify_rms or [float(F["rms"])]
        scores = self.verify_scores or [float(d)]

        post_rms = float(np.median(samples))
        post_score = float(np.median(scores))
        red = 1.0 - post_rms / max(self.rms_before, 1e-9)
        db = 20.0 * np.log10(self.rms_before / max(post_rms, 1e-9))

        # Do not accept a low anomaly score alone as proof that an action worked.
        # A modest RMS reduction can qualify only when the resulting RMS is
        # also close to the healthy baseline.
        baseline_reached = (
            post_score <= self.base.thr
            and post_rms <= self.recovery_rms_limit
        )
        ok = (red >= SUCCESS_REDUCTION) or baseline_reached

        key = (self.fault, self.cur)
        prior = self.mem.get(key, PRIOR.get(self.fault, {}).get(self.cur, 0.5))
        self.mem[key] = 0.6 * prior + 0.4 * float(np.clip(red, 0.0, 1.0))

        if ok:
            self._ev(
                "VERIFY",
                f"{self.cur} mitigation effective: median RMS "
                f"{self.rms_before:.3f} -> {post_rms:.3f} g "
                f"({db:.1f} dB, reduction={100 * red:.1f}%)"
            )
            self.state = "MITIGATED"
            self.mit_rms = post_rms
            self.quiet = self.bad = 0
            self.recovery_elapsed = 0
            self.last_outcome = "MITIGATED_NOT_YET_RECOVERED"
        else:
            self._ev(
                "VERIFY",
                f"{self.cur} mitigation insufficient: median RMS "
                f"{self.rms_before:.3f} -> {post_rms:.3f} g "
                f"({db:.1f} dB, reduction={100 * red:.1f}%)"
            )
            if self.cur == "ACTIVE_REJECT":
                self.rej.disable()
                # The controller is now off, so the machine is back at its original
                # vibration level: judge the next action against the incident RMS,
                # not against the (controller-assisted) post-action RMS.
                ref_rms = self.rms_before
            else:
                # Speed changes are not reverted, so the post-action RMS is the true reference.
                ref_rms = post_rms
            if self.cur is not None:
                self.failed.add(self.cur)
            self._next_action(ref_rms)

    def _safe_stop(self, why):
        self._ev("ACTION", f"SAFE_STOP: {why}")
        self._ev("ALERT", f"maintenance REQUIRED: suspected {self.fault or 'unclassified fault'}; machine stopped")
        self.m.stopped = True; self.rej.disable()
        self.state, self.cur = "STOPPED", "SAFE_STOP"
        self.last_outcome = "STOPPED"
