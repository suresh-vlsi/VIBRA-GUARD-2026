
"""
VIBRA-GUARD
Agentic Edge AI for Intelligent Motor Vibration Monitoring

Dashboard:
1. Motor configuration
2. Single fault simulation
3. Six-fault validation
4. Vibration monitoring
5. FFT analysis
6. Agent diagnosis and mitigation
7. Event logs and CSV exports
"""

import os
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import main as engine

from main import train_baseline
from vibeagent.config import FAULTS, FS, WIN
from vibeagent.dsp import amp_spectrum
from vibeagent.diagnosis import Diagnoser, make_dataset


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VIBRA-GUARD",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs("results", exist_ok=True)

st.title("⚙️ VIBRA-GUARD")
st.subheader(
    "Agentic Edge AI for Intelligent Vibration Monitoring"
)

st.caption(
    "Physics-Based Simulation | DSP | Machine Learning | "
    "Fault Diagnosis | Active Vibration Rejection"
)

st.divider()


# ============================================================
# 2. MODEL INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner="Training AI models...")
def load_models():

    baseline = train_baseline(n=80, seed=5)

    X, y = make_dataset(40, seed=42)

    diagnoser = Diagnoser().fit(X, y)

    return baseline, diagnoser


try:
    with st.spinner("Initializing predictive maintenance engine..."):
        baseline, diagnoser = load_models()

    st.success("AI models initialized successfully.")

except Exception as error:
    st.error(f"Model initialization failed: {error}")
    st.stop()


# ============================================================
# 3. SIMULATION FUNCTION
# ============================================================

def execute_scenario(fault, rpm, windows, injection_window, seed):
    """Run one closed-loop scenario through main.run_scenario."""
    return engine.run_scenario(
        name=fault,
        baseline=baseline,
        diagnoser=diagnoser,
        seed=seed,
        n_win=windows,
        inject_at=injection_window,
        verbose=False,
        rpm=float(rpm),
    )


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Motor Configuration")

st.sidebar.markdown("### Operating parameters")

fault = st.sidebar.selectbox(
    "Select fault condition",
    FAULTS[1:],
    format_func=lambda x: x.replace("_", " ").title()
)

rpm = st.sidebar.slider(
    "Motor speed (RPM)",
    min_value=600,
    max_value=1500,
    value=1200,
    step=50
)

windows = st.sidebar.slider(
    "Simulation windows",
    min_value=20,
    max_value=100,
    value=60
)

inject_at = st.sidebar.slider(
    "Fault injection window",
    min_value=2,
    max_value=windows - 5,
    value=min(10, windows - 5)
)

st.sidebar.divider()

st.sidebar.markdown("### AI detection parameters")

st.sidebar.metric(
    "Healthy baseline RMS",
    f"{baseline.rms0:.4f} g"
)

st.sidebar.metric(
    "Anomaly threshold",
    f"{baseline.thr:.3f}"
)

st.sidebar.metric(
    "Sampling frequency",
    f"{FS} Hz"
)

st.sidebar.divider()

run_single = st.sidebar.button(
    "▶ RUN SELECTED FAULT",
    type="primary",
    width="stretch"
)

run_all = st.sidebar.button(
    "⚡ RUN ALL SIX FAULTS",
    width="stretch"
)


# ============================================================
# 5. RUN SIMULATIONS
# ============================================================

if run_single:

    try:
        with st.spinner(f"Simulating {fault}..."):

            result = execute_scenario(
                fault=fault,
                rpm=rpm,
                windows=windows,
                injection_window=inject_at,
                seed=100
            )

            st.session_state["result"] = result
            st.session_state.pop("batch_results", None)

        st.success("Simulation completed successfully.")

    except Exception as error:
        st.error(f"Simulation failed: {error}")


if run_all:

    results = []

    progress = st.progress(0)

    status = st.empty()

    try:
        for i, fault_name in enumerate(FAULTS[1:]):

            status.write(
                f"Running experiment {i + 1}/6: "
                f"{fault_name.replace('_', ' ').title()}"
            )

            result = execute_scenario(
                fault=fault_name,
                rpm=rpm,
                windows=windows,
                injection_window=inject_at,
                seed=100 + i
            )

            results.append(result)

            progress.progress((i + 1) / 6)

        st.session_state["batch_results"] = results
        st.session_state.pop("result", None)

        status.success("All six experiments completed.")

    except Exception as error:
        status.error(f"Batch execution failed: {error}")


# ============================================================
# 6. COMMON RESULT DISPLAY FUNCTION
# ============================================================

def display_result(result):

    agent = result["agent"]
    events = agent.events
    rms = result["rms"]

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    st.header("1. Motor Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Operating RPM",
        f"{agent.m.rpm:.0f}"
    )

    before, after, db = result["before"], result["after"], result["db"]
    scored = bool(np.isfinite(before) and np.isfinite(after) and before > 0)

    col2.metric("RMS Before", f"{before:.4f} g" if np.isfinite(before) else "N/A")

    if scored:
        col3.metric(
            "RMS After",
            f"{after:.4f} g",
            delta=f"{after - before:.4f} g",
            delta_color="inverse"
        )
        col4.metric("RMS Reduction", f"{100 * (1 - after / before):.1f}%")
    else:
        col3.metric("RMS After", "N/A")
        col4.metric("RMS Reduction", "N/A")

    st.metric(
        "Vibration Reduction",
        f"{db:.2f} dB" if np.isfinite(db) else "NOT SCORED (no verified mitigation)"
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    st.header("2. AI Fault Diagnosis")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Injected Fault",
        result["name"].replace("_", " ").title()
    )

    c2.metric(
        "Diagnosed Fault",
        result["diag"]
    )

    c3.metric(
        "Agent State",
        result["state"]
    )

    recovered = any(
        event["type"] == "RECOVERED"
        for event in events
    )

    stopped = result["state"] == "STOPPED"

    if stopped:

        st.error(
            "SAFE STOP: The simulated machine has been stopped."
        )

    elif recovered:

        st.success(
            "MACHINE RECOVERED: Healthy operating envelope restored."
        )

    elif result["state"] == "MITIGATED":

        st.success(
            "MITIGATION VERIFIED: Agent is monitoring recovery."
        )

    else:

        st.warning(
            "Machine is monitoring or awaiting verified recovery."
        )

    # --------------------------------------------------------
    # VIBRATION TREND
    # --------------------------------------------------------

    st.divider()

    st.header("3. Vibration Monitoring")

    df = pd.DataFrame({
        "Window": np.arange(1, len(rms) + 1),
        "RMS (g)": rms
    })

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        df["Window"],
        df["RMS (g)"],
        label="Measured RMS",
        linewidth=1.5
    )

    ax.axvline(
        result["inject_at"] + 1,
        color="red",
        linestyle="--",
        label="Fault injection"
    )

    for event in events:

        if event["type"] == "ACTION":

            ax.axvline(
                event["win"],
                color="green",
                linestyle=":",
                alpha=0.8
            )

    ax.set_xlabel("Simulation Window")
    ax.set_ylabel("RMS Acceleration (g)")
    ax.set_title("Motor Vibration Trend")

    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    st.divider()

    st.header("4. Frequency Domain Analysis")

    if agent.sigs:

        signal_data = agent.sigs[-1]

        spectrum = amp_spectrum(signal_data)

        frequencies = np.fft.rfftfreq(
            WIN,
            d=1 / FS
        )

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(frequencies, spectrum)

        ax.set_xlim(0, min(500, FS / 2))

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (g)")
        ax.set_title("FFT of Processed Vibration Signal")

        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # PROCESSED SIGNAL
    # --------------------------------------------------------

    st.divider()

    st.header("5. Processed Vibration Signal")

    if agent.sigs:

        signal_data = agent.sigs[-1]

        time_axis = np.arange(len(signal_data)) / FS

        fig, ax = plt.subplots(figsize=(12, 3))

        ax.plot(time_axis, signal_data)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Acceleration (g)")

        ax.set_title(
            "Principal-Axis Vibration After PCA Preprocessing"
        )

        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # AGENT DECISION HISTORY
    # --------------------------------------------------------

    st.divider()

    st.header("6. Autonomous Agent Decision History")

    if events:

        event_df = pd.DataFrame(events)

        st.dataframe(
            event_df[["win", "t", "type", "msg"]],
            width="stretch",
            hide_index=True
        )

    else:

        st.info("No events recorded.")

    # --------------------------------------------------------
    # MITIGATION ACTIONS
    # --------------------------------------------------------

    st.divider()

    st.header("7. Mitigation Actions")

    actions = [
        event["msg"]
        for event in events
        if event["type"] == "ACTION"
    ]

    if actions:

        for action in actions:
            st.write("⚙️", action)

    else:

        st.info("No mitigation actions recorded.")

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.divider()

    st.header("8. Export Engineering Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Vibration Data",
        data=csv,
        file_name=f"{result['name']}_vibration.csv",
        mime="text/csv"
    )

    if events:

        event_csv = pd.DataFrame(events).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Agent Event Log",
            data=event_csv,
            file_name=f"{result['name']}_events.csv",
            mime="text/csv"
        )


# ============================================================
# 7. MAIN DASHBOARD
# ============================================================

if "result" in st.session_state:

    display_result(st.session_state["result"])


elif "batch_results" in st.session_state:

    results = st.session_state["batch_results"]

    st.header("Six-Fault Validation Results")

    summary = pd.DataFrame([
        {
            "Fault": r["name"],
            "Diagnosed": r["diag"],
            "Detection Window": (
                r["det"] if r["det"] is not None else "Not detected"
            ),
            "RMS Before (g)": r["before"],
            "RMS After (g)": r["after"],
            "Reduction (dB)": r["db"],
            "Agent State": r["state"],
            "Actions": " → ".join(r["actions"])
        }
        for r in results
    ])

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("Comparative Fault Mitigation")

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results))
    width = 0.35

    ax.bar(
        x - width / 2,
        [r["before"] for r in results],
        width,
        label="Before mitigation"
    )

    ax.bar(
        x + width / 2,
        [r["after"] for r in results],
        width,
        label="After mitigation"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [r["name"].replace("_", "\n") for r in results]
    )

    ax.set_ylabel("RMS Acceleration (g)")
    ax.set_title("Six-Fault Mitigation Performance")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Individual Experiment Analysis")

    selected = st.selectbox(
        "Select experiment",
        range(len(results)),
        format_func=lambda i: results[i]["name"].replace("_", " ").title()
    )

    display_result(results[selected])

    st.divider()

    st.subheader("Download Batch Report")

    st.download_button(
        "Download Six-Fault Summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="vibra_guard_six_fault_validation.csv",
        mime="text/csv"
    )


else:

    # --------------------------------------------------------
    # INITIAL DASHBOARD
    # --------------------------------------------------------

    st.header("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Fault Classes", "6")
    c2.metric("IMU Channels", "3")
    c3.metric("Sampling Rate", f"{FS} Hz")
    c4.metric("Analysis Window", f"{WIN} samples")

    st.divider()

    st.header("System Architecture")

    st.code(
        """
        THREE-AXIS IMU
              |
              v
        SIGNAL PREPROCESSING
        (High-pass + PCA)
              |
              v
        FEATURE EXTRACTION
        (FFT + Envelope + RMS)
              |
              v
        ANOMALY DETECTION
        (Mahalanobis + EWMA)
              |
              v
        FAULT DIAGNOSIS
        (Random Forest + Physics)
              |
              v
        AGENT DECISION ENGINE
              |
              v
        MITIGATION PLANNING
              |
              v
        ACTIVE VIBRATION REJECTION
        (FxLMS / Speed Shift / Derating)
              |
              v
        VERIFICATION + ONLINE LEARNING
        """,
        language="text"
    )

    st.info(
        "Select a fault from the sidebar and click "
        "RUN SELECTED FAULT, or execute all six experiments."
    )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "VIBRA-GUARD | Agentic Predictive Maintenance Research Prototype"
)

st.caption(
    "Simulation-based results. Hardware validation and "
    "industrial safety certification are not implied."
)
