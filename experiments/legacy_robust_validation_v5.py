import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

"""
VIBRA-GUARD V5 — randomized simulation validation.

Run from the project root:
    python robust_validation.py --trials 5
    python robust_validation.py --trials 20 --quiet

This script does not modify main.py or vibeagent/agent.py.
It imports the currently installed Agent, so ensure vibeagent/agent.py is V4.
"""
import argparse
import csv
import os
import time

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from vibeagent.config import (
    FAULTS, FN_HEALTHY, RPM_MIN, RPM_MAX, RPM_NOM, WIN
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
    """Train a healthy baseline and fault diagnoser for this validation run."""
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


def run_trial(label, trial_id, seed, baseline, diagnoser,
              windows=60, inject_at=10, verbose=False):
    """
    Run one randomized scenario.

    Healthy trials use the nominal structural mode. Fault trials randomize
    RPM and severity. Resonance trials randomize the structural mode around
    the second shaft order, without passing the true mode to the agent.
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

    machine = Machine(rpm=rpm, fn=mode_hz, seed=seed)
    rejector = ActiveRejector()
    agent = Agent(
        machine, baseline, diagnoser, rejector,
        log_path=None, verbose=verbose
    )
    agent.calibrate()

    # Each trial gets its own baseline copy: online learning cannot leak
    # from one randomized trial into another.
    import copy
    agent.base = copy.deepcopy(baseline)

    for window_index in range(windows):
        if label != "healthy" and window_index == inject_at:
            machine.inject(injected_fault, injected_severity)
        agent.tick(truth=label if window_index >= inject_at else "healthy")

    anomaly_event = first_event(agent, "ANOMALY")
    diagnosis_event = first_event(agent, "DIAGNOSE")
    verify_events = [
        event for event in agent.events if event.get("type") == "VERIFY"
    ]
    effective_verify = any(
        "mitigation effective" in event.get("msg", "").lower()
        and "insufficient" not in event.get("msg", "").lower()
        for event in verify_events
    )
    stopped = agent.state == "STOPPED" or any(
        "SAFE_STOP" in event.get("msg", "") for event in agent.events
    )
    recovered = first_event(agent, "RECOVERED") is not None

    diagnosed = "none"
    if diagnosis_event:
        diagnosed = diagnosis_event.get("msg", "unknown").split(" (")[0]

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
    if anomaly_event:
        detection_latency = max(
            0, int(anomaly_event.get("win", 0)) - inject_at - 1
        )

    return {
        "trial_id": trial_id,
        "seed": seed,
        "fault": label,
        "rpm_start": round(rpm, 2),
        "severity": round(severity, 3),
        "structural_mode_hz": round(mode_hz, 2),
        "detected": anomaly_event is not None,
        "detection_latency_windows": detection_latency,
        "diagnosed": diagnosed,
        "diagnosis_correct": correct_diagnosis,
        "mitigation_verified": effective_verify,
        "recovered": recovered,
        "safe_stop": stopped,
        "false_alarm": false_alarm,
        "outcome": outcome,
        "final_state": agent.state,
        "actions": ";".join(
            event.get("msg", "").split(":")[0]
            for event in agent.events
            if event.get("type") == "ACTION"
        ),
    }


def summarize(rows):
    healthy = [row for row in rows if row["fault"] == "healthy"]
    faulty = [row for row in rows if row["fault"] != "healthy"]

    print("\n" + "=" * 76)
    print("V5 RANDOMIZED VALIDATION SUMMARY")
    print("=" * 76)
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
        print(f"Fault detection rate:         {100 * detection:.1f}%")
        print(f"Correct diagnosis rate:       {100 * diagnosis:.1f}%")
        print(f"Mitigation verification rate: {100 * verified:.1f}%")
        print(f"Recovery rate:                {100 * recovery:.1f}%")
        print(f"Safety-stop rate:             {100 * stops:.1f}%")

        detected_latencies = [
            row["detection_latency_windows"]
            for row in faulty
            if row["detection_latency_windows"] is not None
        ]
        if detected_latencies:
            print(
                "Median detection latency:    "
                f"{np.median(detected_latencies):.1f} windows"
            )

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
        labels = FAULT_CLASSES
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        print("\nFault diagnosis confusion matrix (rows=true, columns=predicted)")
        print("Class order:", labels)
        print(matrix)

    print("\nInterpretation:")
    print("  SAFE_STOP is a safety response, not a mitigation success.")
    print("  MITIGATED_NOT_RECOVERED means vibration reduction was verified,")
    print("  but return to the healthy envelope was not established.")
    print("  Rates are estimates for this simulator and this chosen test matrix.")


def main():
    parser = argparse.ArgumentParser(
        description="VIBRA-GUARD V5 randomized robustness validation"
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Number of trials per class, including healthy (default: 5)."
    )
    parser.add_argument(
        "--windows", type=int, default=60,
        help="Agent windows per trial (default: 60)."
    )
    parser.add_argument(
        "--inject-at", type=int, default=10,
        help="Zero-based fault injection window (default: 10)."
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Master random seed (default: 2026)."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-window agent messages."
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.windows <= args.inject_at:
        parser.error("--windows must be greater than --inject-at")

    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()

    print("VIBRA-GUARD V5 — RANDOMIZED SIMULATION VALIDATION")
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
            result = run_trial(
                label, trial_number, trial_seed, baseline, diagnoser,
                windows=args.windows,
                inject_at=args.inject_at,
                verbose=not args.quiet,
            )
            rows.append(result)

    output_path = os.path.join(OUT_DIR, "v5_randomized_validation.csv")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nTrial-level results saved to: {output_path}")
    print(f"Total runtime: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
