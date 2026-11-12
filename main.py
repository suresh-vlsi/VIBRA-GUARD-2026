"""VIBRA-GUARD experiment runner.

Run from the project root:
    python main.py
    python main.py --quick
    python main.py --quick --quiet

This runner separates verified, running-machine mitigation from SAFE_STOP.
A stopped machine is never credited with vibration-reduction performance.
"""
import argparse
import copy
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from vibeagent.config import *
from vibeagent.simulator import Machine
from vibeagent.dsp import preprocess, features, amp_spectrum
from vibeagent.anomaly import HealthBaseline
from vibeagent.diagnosis import Diagnoser, make_dataset
from vibeagent.control import ActiveRejector
from vibeagent.agent import Agent


OUT = "results"


def train_baseline(n=150, seed=5):
    """Fit the unsupervised healthy reference over a range of shaft speeds."""
    rng = np.random.default_rng(seed)
    machine = Machine(fn=FN_HEALTHY, seed=seed)
    samples = []

    for _ in range(n):
        machine.set_rpm(rng.uniform(900, 1500))
        machine.step(WIN)
        signal_window = machine.step(WIN)
        samples.append(features(preprocess(signal_window), machine.rpm, FN_HEALTHY))

    return HealthBaseline().fit(samples)


def envelope_window(fault, rng, seed):
    """Generate a labelled sample inside the normal operating envelope."""
    rpm = rng.uniform(900, 1500)

    if fault == "resonance":
        fn = 2.0 * rpm / 60.0 * (1.0 + rng.uniform(-0.03, 0.03))
        severity, injected_fault = 0.0, "healthy"
    else:
        fn = FN_HEALTHY
        severity, injected_fault = rng.uniform(0.4, 1.0), fault

    machine = Machine(rpm=rpm, fn=fn, seed=seed)
    machine.inject(injected_fault, severity)
    machine.step(WIN)
    signal_window = machine.step(WIN)
    return features(preprocess(signal_window), rpm, fn)


def evaluate(baseline, diagnoser, n=40):
    """Evaluate diagnosis, anomaly detection and fused diagnosis on held-out data."""
    rng = np.random.default_rng(999)

    # Diagnosis model on a separately generated unseen dataset.
    x_test, y_test = make_dataset(n, seed=77)
    y_pred = diagnoser.rf.predict(x_test)
    print("\nDIAGNOSIS MODEL — UNSEEN SIMULATED DATA")
    print(f"Random-forest accuracy: {np.mean(y_pred == y_test) * 100:.1f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Healthy false-alarm rate and fault detection recall.
    healthy = [
        envelope_window("healthy", rng, 5000 + i)
        for i in range(200)
    ]
    faulty, fault_labels = [], []
    for class_index, fault in enumerate(FAULTS[1:], start=1):
        for i in range(n):
            faulty.append(
                envelope_window(
                    fault, rng, 9000 + class_index * 100 + i
                )
            )
            fault_labels.append(fault)

    healthy_detected = np.array(
        [baseline.score(sample)[0] > baseline.thr for sample in healthy]
    )
    faulty_detected = np.array(
        [baseline.score(sample)[0] > baseline.thr for sample in faulty]
    )

    y_true = np.r_[
        np.zeros(len(healthy_detected), dtype=int),
        np.ones(len(faulty_detected), dtype=int),
    ]
    y_detection = np.r_[healthy_detected, faulty_detected]

    print("ANOMALY DETECTION — OPERATING ENVELOPE")
    print(f"F1 score:          {f1_score(y_true, y_detection, zero_division=0):.3f}")
    print(f"Healthy false alarm rate: {healthy_detected.mean() * 100:.1f}%")
    print(f"Fault detection recall:   {faulty_detected.mean() * 100:.1f}%")
    for fault in FAULTS[1:]:
        selected = np.array(fault_labels) == fault
        rate = faulty_detected[selected].mean() if selected.any() else float("nan")
        print(f"  {fault:<16}: {rate * 100:5.1f}%")

    # Rule + RF fused diagnosis. Include all classes in the confusion matrix.
    true_labels, predicted_labels = [], []
    for class_index, fault in enumerate(FAULTS[1:], start=1):
        for i in range(n):
            sample = envelope_window(
                fault, rng, 20000 + i + 100 * class_index
            )
            probabilities, _ = diagnoser.probs(sample)
            true_labels.append(class_index)
            predicted_labels.append(int(np.argmax(probabilities)))

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=range(len(FAULTS))
    )
    fused_accuracy = np.mean(
        np.asarray(true_labels) == np.asarray(predicted_labels)
    )
    print("\nFUSED DIAGNOSIS — OPERATING-ENVELOPE FAULTS")
    print(f"Accuracy: {fused_accuracy * 100:.1f}%")
    print("Confusion matrix (rows=true, columns=predicted)")
    print("Class order:", FAULTS)
    print(matrix)


def _event_first(agent, event_type):
    return next((event for event in agent.events
                 if event.get("type") == event_type), None)


def _event_actions(agent):
    return [
        event.get("msg", "").split(":")[0]
        for event in agent.events
        if event.get("type") == "ACTION"
    ]


def run_scenario(name, baseline, diagnoser, seed,
                 n_win=60, inject_at=10, verbose=True, rpm=None):
    """Run one closed-loop fault scenario and score only verified active operation.

    ``rpm`` is the starting shaft speed (default: RPM_NOM).  For the resonance
    scenario the structural mode is moved to 2x the shaft frequency at that speed,
    so the scenario is genuinely resonant at any ``rpm``.
    """
    os.makedirs(OUT, exist_ok=True)
    rpm = float(RPM_NOM if rpm is None else rpm)

    machine = Machine(rpm=rpm, fn=FN_HEALTHY, seed=seed)
    rejector = ActiveRejector()
    agent = Agent(
        machine,
        baseline,
        diagnoser,
        rejector,
        log_path=os.path.join(OUT, f"events_{name}.jsonl"),
        verbose=verbose,
    )

    agent.calibrate()
    # Keep scenarios independent: online baseline learning in one scenario
    # must not alter the baseline used by another.
    agent.base = copy.deepcopy(baseline)

    for window_index in range(n_win):
        if window_index == inject_at:
            if name == "resonance":
                machine.set_fn(2.0 * rpm / 60.0)
            else:
                machine.inject(name, 0.8)
            if verbose:
                print(f"  >>> injecting fault: {name}")

        agent.tick(truth=name if window_index >= inject_at else "healthy")

    rows = agent.rows
    first_fault_win = inject_at + 1          # first window containing the fault
    post_events = [e for e in agent.events if e["win"] >= first_fault_win]
    rms = np.asarray([row["rms"] for row in rows], dtype=float)
    rpm = np.asarray([row["rpm"] for row in rows], dtype=float)

    action_event = next(
        (event for event in agent.events if event.get("type") == "ACTION"),
        None,
    )
    verify_events = [
        event for event in agent.events if event.get("type") == "VERIFY"
    ]
    verify_event = verify_events[-1] if verify_events else None
    recovered_event = _event_first(agent, "RECOVERED")

    # Agent verification compares its own pre-action reference (usually the
    # mean RMS of anomalous diagnosis windows) with post-action verification
    # windows. That reference is not necessarily the RMS in the action row.
    # Parse the exact reference used by Agent so the summary agrees with VERIFY.
    if action_event is not None:
        action_win = int(action_event["win"])
    else:
        action_win = None

    verify_message_raw = verify_event.get("msg", "") if verify_event else ""
    verify_match = re.search(
        r"median RMS\s+([0-9]+(?:\.[0-9]+)?)\s*->\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*g",
        verify_message_raw,
        flags=re.IGNORECASE,
    )

    if verify_match:
        before = float(verify_match.group(1))
    elif action_event is not None:
        action_rows = [row["rms"] for row in rows
                       if row["win"] == action_win]
        before = float(np.median(action_rows)) if action_rows else float("nan")
    else:
        before = float("nan")

    verify_message = verify_message_raw.lower()
    mitigation_verified = (
        verify_event is not None
        and "mitigation effective" in verify_message
        and "insufficient" not in verify_message
    )

    # Only use post-verification windows while the machine is still rotating.
    # Exclude the verification window itself: its row was recorded before
    # Agent.tick changed the state from VERIFY to MITIGATED.
    verify_win = int(verify_event["win"]) if verify_event else -1
    active_post_rows = [
        row for row in rows
        if row["win"] > verify_win
        and row["rpm"] > 0
        and row["state"] != "STOPPED"
    ]

    if mitigation_verified and agent.state != "STOPPED" and active_post_rows:
        after_values = [row["rms"] for row in active_post_rows[-10:]]
        after = float(np.median(after_values))
        reduction_db = (
            20.0 * np.log10(before / max(after, 1e-12))
            if np.isfinite(before) and before > 0 else float("nan")
        )
        outcome = (
            "RECOVERED" if recovered_event is not None
            else "MITIGATED_NOT_RECOVERED"
        )
    else:
        # Keep a diagnostic post-action RMS if available, but do not report
        # a dB reduction as a successful mitigation result.
        after = (
            float(np.median([row["rms"] for row in active_post_rows[-10:]]))
            if active_post_rows else float("nan")
        )
        reduction_db = float("nan")
        if agent.state == "STOPPED":
            outcome = "STOPPED"
        elif verify_event is not None:
            outcome = "MITIGATION_FAILED"
        else:
            outcome = "NOT_VERIFIED"

    diagnosed = "-"
    diagnosis_event = next((e for e in post_events if e["type"] == "DIAGNOSE"), None)
    if diagnosis_event:
        diagnosed = diagnosis_event.get("msg", "-").split(" (")[0]

    return {
        "name": name,
        "agent": agent,
        "rows": rows,
        "rms": rms,
        "rpm": rpm,
        "det": next(
            (event["win"] - first_fault_win
             for event in post_events if event.get("type") == "ANOMALY"),
            None,
        ),
        "before": before,
        "after": after,
        "db": reduction_db,
        "state": agent.state,
        "outcome": outcome,
        "verified": mitigation_verified,
        "diag": diagnosed,
        "actions": _event_actions(agent),
        "inject_at": inject_at,
        "action_win": action_win,
        "verify_win": verify_win,
    }


def make_plots(results):
    """Save scenario trends and a representative before/after spectrum."""
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for index, ax in enumerate(axes):
        if index >= len(results):
            ax.axis("off")
            continue

        result = results[index]
        rows = result["rows"]
        windows = np.asarray([row["win"] for row in rows])
        rms = np.asarray([row["rms"] for row in rows], dtype=float)
        speeds = np.asarray([row["rpm"] for row in rows], dtype=float)

        # Do not let shutdown's near-zero RMS visually imply mitigation.
        rms_plot = rms.copy()
        rms_plot[speeds <= 0] = np.nan

        ax.plot(windows, rms_plot, lw=1.3, label="RMS while running")
        ax.axvline(
            result["inject_at"] + 1, color="red", ls="--",
            lw=1.0, label="fault injection",
        )

        for event in result["agent"].events:
            if event.get("type") == "ACTION":
                ax.axvline(event["win"], color="green", ls=":", lw=1.0)

        ax.set_title(
            f"{result['name']} | {result['outcome']}",
            fontsize=9,
        )
        ax.set_ylabel("RMS (g)")
        ax.grid(alpha=0.3)

    axes[-2].set_xlabel("Agent window")
    axes[-1].set_xlabel("Agent window")
    axes[0].legend(fontsize=7)
    fig.suptitle("VIBRA-GUARD closed-loop fault scenarios")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scenarios.png"), dpi=150)
    plt.close(fig)

    # Imbalance spectrum: compare only when there is a valid active-running
    # post-verification interval.
    imbalance = next(
        (result for result in results if result["name"] == "imbalance"),
        None,
    )
    if imbalance is None:
        return

    agent = imbalance["agent"]
    if not agent.sigs:
        return

    before_index = max(0, min(len(agent.sigs) - 1,
                             (imbalance["action_win"] or 1) - 1))
    after_index = max(0, min(len(agent.sigs) - 1,
                            len(agent.sigs) - 1))

    frequency = np.fft.rfftfreq(WIN, d=1.0 / FS)
    before_spectrum = amp_spectrum(agent.sigs[before_index])

    plt.figure(figsize=(9, 4.5))
    plt.semilogy(frequency, before_spectrum, label="Before action")

    if imbalance["verified"] and imbalance["state"] != "STOPPED":
        after_spectrum = amp_spectrum(agent.sigs[after_index])
        plt.semilogy(frequency, after_spectrum, label="After action")

    plt.xlim(0, 200)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (g)")
    plt.title("Imbalance spectrum — active-operation comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "spectrum_imbalance.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run VIBRA-GUARD predictive-maintenance experiments."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller training/evaluation datasets.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-window agent messages.")
    parser.add_argument("--windows", type=int, default=60,
                        help="Number of windows per fault scenario.")
    parser.add_argument("--inject-at", type=int, default=10,
                        help="Zero-based window index at which to inject the fault.")
    args = parser.parse_args()

    if args.windows <= args.inject_at:
        parser.error("--windows must be greater than --inject-at")

    os.makedirs(OUT, exist_ok=True)
    start_time = time.time()

    print("=" * 76)
    print("VIBRA-GUARD | PHYSICS-INFORMED AGENTIC VIBRATION MONITORING")
    print("=" * 76)

    print("\n[1/4] Training healthy baseline...")
    baseline = train_baseline(80 if args.quick else 150)
    print(f"Healthy baseline RMS: {baseline.rms0:.4f} g")
    print(f"Anomaly threshold:    {baseline.thr:.4f}")

    print("\n[2/4] Training fault-diagnosis model...")
    x_train, y_train = make_dataset(40 if args.quick else 120)
    diagnoser = Diagnoser().fit(x_train, y_train)
    print(f"Training samples: {len(y_train)}")

    print("\n[3/4] Evaluating detection and diagnosis...")
    evaluate(baseline, diagnoser, n=20 if args.quick else 40)

    print("\n[4/4] Running closed-loop fault scenarios...")
    results = []
    for index, fault in enumerate(FAULTS[1:]):
        print(f"\n--- Scenario: {fault} ---")
        result = run_scenario(
            fault,
            baseline,
            diagnoser,
            seed=100 + index,
            n_win=args.windows,
            inject_at=args.inject_at,
            verbose=not args.quiet,
        )
        results.append(result)

    make_plots(results)

    print("\n" + "=" * 112)
    print("CLOSED-LOOP RESULTS")
    print("=" * 112)
    header = (
        f"{'Fault':<16}{'Diagnosis':<16}{'Outcome':<27}"
        f"{'Verify ref':>12}{'Sustained':>12}{'Change':>13}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        after_text = (
            f"{result['after']:.4f}"
            if np.isfinite(result["after"]) else "N/A"
        )
        before_text = (
            f"{result['before']:.4f}"
            if np.isfinite(result["before"]) else "N/A"
        )
        reduction_text = (
            f"{result['db']:.2f} dB"
            if np.isfinite(result["db"]) else "NOT SCORED"
        )
        print(
            f"{result['name']:<16}{result['diag']:<16}"
            f"{result['outcome']:<27}{before_text:>12}"
            f"{after_text:>12}{reduction_text:>13}"
        )

    print("\nInterpretation:")
    print("  RECOVERED              = mitigation followed by verified recovery.")
    print("  MITIGATED_NOT_RECOVERED = action passed mitigation verification,")
    print("                            but healthy recovery was not established.")
    print("  STOPPED                = safety shutdown; never counted as mitigation.")
    print("  NOT SCORED             = no verified active-running reduction.")

    print(f"\nPlots and JSONL event logs saved in: ./{OUT}/")
    print(f"Total runtime: {time.time() - start_time:.1f} s")


if __name__ == "__main__":
    main()
