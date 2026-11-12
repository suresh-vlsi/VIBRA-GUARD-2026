"""
VIBRA-GUARD V5.1 — corrected randomized simulation validation.

Run from the project root:
    python robust_validation_v5_1.py --trials 20 --quiet

This script does not modify main.py or vibeagent/agent.py.
It imports the currently installed Agent.

Important experimental correction:
All machines start with the healthy structural mode. For resonance trials,
the structural mode changes only at the fault-injection window, AFTER
healthy calibration. The agent is not given the true mode.
"""
import argparse
import copy
import csv
import os
import re
import time

import numpy as np
from sklearn.metrics import confusion_matrix

from vibeagent.config import (
    FAULTS, FN_HEALTHY, RPM_NOM, WIN
)
from vibeagent.simulator import Machine
from vibeagent.dsp import preprocess, features
from vibeagent.anomaly import HealthBaseline
from vibeagent.diagnosis import Diagnoser, make_dataset
from vibeagent.control import ActiveRejector
from vibeagent.agent import Agent


OUT_DIR = "results"
FAULT_CLASSES = list(FAULTS[1:])


def train_models(seed=510):
    """Train a healthy baseline and fault diagnoser."""
    rng = np.random.default_rng(seed)
    machine = Machine(rpm=RPM_NOM, fn=FN_HEALTHY, seed=seed)
    healthy_features = []

    for _ in range(120):
        rpm = float(rng.uniform(900, 1500))
        machine.set_rpm(rpm)
        machine.step(WIN)
        raw = machine.step(WIN)
        healthy_features.append(
            features(preprocess(raw), machine.rpm, FN_HEALTHY)
        )

    baseline = HealthBaseline().fit(healthy_features)
    x_train, y_train = make_dataset(60, seed=seed + 1)
    diagnoser = Diagnoser().fit(x_train, y_train)
    return baseline, diagnoser


def first_event(agent, event_type):
    return next(
        (event for event in agent.events if event.get("type") == event_type),
        None,
    )


def parse_number(pattern, text):
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def run_trial(label, trial_id, seed, baseline, diagnoser,
              windows=60, inject_at=10, verbose=False):
    """
    Run one randomized scenario.

    Every machine starts healthy at FN_HEALTHY and is calibrated in that
    condition. Faults are injected at inject_at. Resonance is introduced
    by changing the structural mode at that same instant.
    """
    rng = np.random.default_rng(seed)
    rpm = float(rng.uniform(900, 1500))
    severity = 0.0 if label == "healthy" else float(rng.uniform(0.55, 1.0))

    if label == "resonance":
        mode_hz = (2.0 * rpm / 60.0) * float(rng.uniform(0.97, 1.03))
        injected_fault = "healthy"
        injected_severity = 0.0
    else:
        mode_hz = FN_HEALTHY
        injected_fault = label
        injected_severity = severity

    # Critical correction: initialize at the healthy mode for EVERY trial.
    machine = Machine(rpm=rpm, fn=FN_HEALTHY, seed=seed)
    rejector = ActiveRejector()
    agent = Agent(
        machine, baseline, diagnoser, rejector,
        log_path=None, verbose=verbose
    )

    # Calibration is completed before any fault is introduced.
    agent.calibrate()
    agent.base = copy.deepcopy(baseline)

    for window_index in range(windows):
        if label != "healthy" and window_index == inject_at:
            if label == "resonance":
                machine.set_fn(mode_hz)
            machine.inject(injected_fault, injected_severity)

        agent.tick(
            truth=label if window_index >= inject_at else "healthy"
        )

    # A fault trial can only be "detected"/"diagnosed" by events AFTER injection;
    # a pre-injection false alarm must not be credited as a detection.
    if label == "healthy":
        scored_events = agent.events
    else:
        scored_events = [e for e in agent.events if e["win"] > inject_at]
    anomaly_event = next((e for e in scored_events if e["type"] == "ANOMALY"), None)
    diagnosis_event = next((e for e in scored_events if e["type"] == "DIAGNOSE"), None)
    trip_after_injection = next(
        (e for e in scored_events if e["type"] == "TRIP"), None
    )
    pre_injection_alarm = (
        label != "healthy"
        and any(e["type"] == "ANOMALY" and e["win"] <= inject_at for e in agent.events)
    )
    verify_events = [
        event for event in agent.events if event.get("type") == "VERIFY"
    ]
    effective_verify = any(
        "mitigation effective" in event.get("msg", "").lower()
        and "insufficient" not in event.get("msg", "").lower()
        for event in verify_events
    )

    trip_event = first_event(agent, "TRIP")
    safe_stop_event = next(
        (
            event for event in agent.events
            if event.get("type") == "ACTION"
            and "SAFE_STOP" in event.get("msg", "")
        ),
        None,
    )
    stopped = agent.state == "STOPPED" or safe_stop_event is not None
    recovered = first_event(agent, "RECOVERED") is not None

    diagnosed = "none"
    diagnosis_confidence = None
    diagnosis_severity = None
    incident_rms_g = None
    if diagnosis_event:
        message = diagnosis_event.get("msg", "")
        diagnosed = message.split(" (")[0]
        diagnosis_confidence = parse_number(r"conf\s+([0-9.]+)", message)
        severity_match = re.search(r"severity\s+([a-zA-Z_]+)", message)
        diagnosis_severity = severity_match.group(1) if severity_match else None
        incident_rms_g = parse_number(r"RMS\s+([0-9.]+)\s*g", message)

    if label == "healthy":
        false_alarm = anomaly_event is not None
        outcome = "FALSE_ALARM" if false_alarm else "HEALTHY_NO_ALARM"
        correct_diagnosis = not false_alarm
    else:
        false_alarm = False
        correct_diagnosis = diagnosed == label
        if stopped:
            outcome = "SAFE_STOP"
        elif recovered:
            outcome = "RECOVERED"
        elif effective_verify:
            outcome = "MITIGATED_NOT_RECOVERED"
        elif verify_events:
            outcome = "MITIGATION_FAILED"
        else:
            outcome = "NOT_VERIFIED"

    detection_latency = None
    first_detection = anomaly_event or trip_after_injection
    if first_detection:
        detection_latency = max(
            0, int(first_detection.get("win", 0)) - inject_at - 1
        )

    trip_rms_g = None
    trip_stage = None
    trip_state = None
    if trip_event:
        trip_rms_g = parse_number(r"RMS\s+([0-9.]+)\s*g", trip_event.get("msg", ""))
        trip_win = trip_event.get("win")
        matching_rows = [
            row for row in agent.rows if row.get("win") == trip_win
        ]
        if matching_rows:
            trip_state = matching_rows[-1].get("state")

        # A TRIP can occur inside a nested PROBE/ACTION routine while the
        # outer state-machine state has not yet changed. Record the preceding
        # event as the operational context instead of mislabeling it MONITOR.
        trip_index = agent.events.index(trip_event)
        if trip_index > 0:
            previous_event = agent.events[trip_index - 1]
            trip_stage = previous_event.get("type")

    stop_reason = None
    if safe_stop_event:
        stop_reason = safe_stop_event.get("msg", "")

    actions = ";".join(
        event.get("msg", "").split(":")[0]
        for event in agent.events
        if event.get("type") == "ACTION"
    )

    return {
        "trial_id": trial_id,
        "seed": seed,
        "fault": label,
        "rpm_start": round(rpm, 2),
        "severity": round(severity, 3),
        "structural_mode_hz": round(mode_hz, 2),
        # A hard-limit TRIP is also a detection (the fault was caught, just by the
        # safety layer before the anomaly detector had a chance to report).
        "detected": anomaly_event is not None or trip_after_injection is not None,
        "detected_by_trip_only": anomaly_event is None and trip_after_injection is not None,
        "detection_latency_windows": detection_latency,
        "diagnosed": diagnosed,
        "diagnosis_confidence": diagnosis_confidence,
        "diagnosis_severity": diagnosis_severity,
        "incident_rms_g": incident_rms_g,
        "diagnosis_correct": correct_diagnosis,
        "mitigation_verified": effective_verify,
        "recovered": recovered,
        "safe_stop": stopped,
        "trip_rms_g": trip_rms_g,
        "trip_stage": trip_stage,
        "trip_state": trip_state,
        "stop_reason": stop_reason,
        "false_alarm": false_alarm or pre_injection_alarm,
        "outcome": outcome,
        "final_state": agent.state,
        "actions": actions,
    }


def summarize(rows):
    healthy = [row for row in rows if row["fault"] == "healthy"]
    faulty = [row for row in rows if row["fault"] != "healthy"]

    print("\n" + "=" * 82)
    print("V5.1 CORRECTED RANDOMIZED VALIDATION SUMMARY")
    print("=" * 82)
    print(f"Total trials:                 {len(rows)}")
    print(f"Healthy trials:               {len(healthy)}")
    print(f"Fault trials:                 {len(faulty)}")

    if healthy:
        far = np.mean([row["false_alarm"] for row in healthy])
        print(f"Healthy false-alarm rate:     {100 * far:.1f}%")

    if faulty:
        detection = np.mean([row["detected"] for row in faulty])
        diagnosis = np.mean([row["diagnosis_correct"] for row in faulty])
        verified = np.mean([row["mitigation_verified"] for row in faulty])
        recovery = np.mean([row["recovered"] for row in faulty])
        stops = np.mean([row["safe_stop"] for row in faulty])
        pre_alarms = sum(row["false_alarm"] for row in faulty)
        print(f"Pre-injection false alarms in fault trials: {pre_alarms} (not counted as detections)")
        trip_only = sum(row["detected_by_trip_only"] for row in faulty)
        print(f"Fault detection rate:         {100 * detection:.1f}%  "
              f"({trip_only} caught by the hard-limit trip before the anomaly detector)")
        print(f"Correct diagnosis rate:       {100 * diagnosis:.1f}%")
        print(f"Mitigation verification rate: {100 * verified:.1f}%")
        print(f"Recovery rate:                {100 * recovery:.1f}%")
        print(f"Safety-stop rate:             {100 * stops:.1f}%")

        latencies = [
            row["detection_latency_windows"]
            for row in faulty
            if row["detection_latency_windows"] is not None
        ]
        if latencies:
            print(f"Median detection latency:     {np.median(latencies):.1f} windows")

        print("\nPer-fault results:")
        print(
            f"{'Fault':<16}{'N':>5}{'Detect':>10}"
            f"{'Diagnose':>11}{'Verify':>10}{'Recover':>10}{'Stop':>8}"
        )
        for fault in FAULT_CLASSES:
            group = [row for row in faulty if row["fault"] == fault]
            if not group:
                continue
            n = len(group)
            print(
                f"{fault:<16}{n:>5}"
                f"{100*np.mean([x['detected'] for x in group]):>9.1f}%"
                f"{100*np.mean([x['diagnosis_correct'] for x in group]):>10.1f}%"
                f"{100*np.mean([x['mitigation_verified'] for x in group]):>9.1f}%"
                f"{100*np.mean([x['recovered'] for x in group]):>9.1f}%"
                f"{100*np.mean([x['safe_stop'] for x in group]):>7.1f}%"
            )

        y_true = [row["fault"] for row in faulty]
        y_pred = [row["diagnosed"] for row in faulty]
        print("\nFault diagnosis confusion matrix (rows=true, columns=predicted)")
        print("Class order:", FAULT_CLASSES)
        print(confusion_matrix(y_true, y_pred, labels=FAULT_CLASSES))

        print("\nSafety-stop breakdown:")
        stopped = [row for row in faulty if row["safe_stop"]]
        pre_detection = sum(row["detected_by_trip_only"] for row in stopped)
        after_detection = len(stopped) - pre_detection
        print(f"Stops by hard-limit trip before anomaly detection: {pre_detection}")
        print(f"Stops after anomaly detection:  {after_detection}")
        print("\nSafety-stop reasons:")
        reason_counts = {}
        for row in stopped:
            reason = row["stop_reason"] or "TRIP_WITHOUT_SAFE_STOP_EVENT"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:>3}  {reason}")

    print("\nInterpretation:")
    print("  SAFE_STOP is a safety response, not a mitigation success.")
    print("  Resonance now begins from a healthy 70 Hz plant and changes at injection.")
    print("  Rates describe this simulator and this randomized test distribution.")


def main():
    parser = argparse.ArgumentParser(
        description="VIBRA-GUARD V5.1 corrected randomized validation"
    )
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per class, including healthy (default: 5).")
    parser.add_argument("--windows", type=int, default=60,
                        help="Agent windows per trial (default: 60).")
    parser.add_argument("--inject-at", type=int, default=10,
                        help="Zero-based fault injection window (default: 10).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Master random seed (default: 2026).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-window agent messages.")
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.windows <= args.inject_at:
        parser.error("--windows must be greater than --inject-at")
    if args.inject_at < 0:
        parser.error("--inject-at must be non-negative")

    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()

    print("VIBRA-GUARD V5.1 — CORRECTED RANDOMIZED VALIDATION")
    print(f"Trials/class: {args.trials} | windows/trial: {args.windows}")
    print("Training validation models...")
    baseline, diagnoser = train_models(seed=args.seed)
    print(
        f"Healthy baseline RMS={baseline.rms0:.4f} g | "
        f"threshold={baseline.thr:.4f}"
    )

    rows = []
    classes = ["healthy"] + FAULT_CLASSES
    trial_number = 0
    for class_index, label in enumerate(classes):
        for replicate in range(args.trials):
            trial_number += 1
            trial_seed = args.seed + 1000 * class_index + replicate
            print(
                f"[{trial_number:03d}/{len(classes)*args.trials:03d}] "
                f"{label} trial {replicate + 1}/{args.trials}",
                flush=True,
            )
            rows.append(
                run_trial(
                    label, trial_number, trial_seed, baseline, diagnoser,
                    windows=args.windows,
                    inject_at=args.inject_at,
                    verbose=not args.quiet,
                )
            )

    output_path = os.path.join(OUT_DIR, "v5_1_corrected_validation.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nTrial-level results saved to: {output_path}")
    print(f"Total runtime: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
