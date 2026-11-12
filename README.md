# VIBRA-GUARD
### Agentic Vibration Monitoring, Fault Diagnosis & Active Rejection

<p align="center">
  <a href="https://suresh-vlsi.github.io/VIBRA-GUARD/">
    <img src="https://img.shields.io/badge/Project-Portfolio-36D7C2?style=for-the-badge" />
  </a>
  <a href="https://vibra-guard.streamlit.app/">
    <img src="https://img.shields.io/badge/Live-Interactive_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="https://github.com/suresh-vlsi/VIBRA-GUARD">
    <img src="https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github" />
  </a>
</p>

**Explore the project**

- 🌐 [Official Project Portfolio](https://suresh-vlsi.github.io/VIBRA-GUARD/)
- 🚀 [Launch Interactive Dashboard](https://vibra-guard.streamlit.app/)
- 💻 [View Source Code](https://github.com/suresh-vlsi/VIBRA-GUARD)

---

## 1. Project Overview

VIBRA-GUARD is a **software-based, simulation-driven demonstrator** for intelligent vibration monitoring, fault diagnosis and active vibration rejection in rotating machinery.

The system integrates a physics-based machine simulator, synthetic three-axis IMU signals, digital signal processing, machine learning, an agentic decision-making architecture and closed-loop control.

Its objective is to demonstrate how an intelligent monitoring agent can detect abnormal machine behaviour, investigate potential faults, select mitigation strategies and independently verify their outcomes.

> **Validation scope:** All reported results are simulation-based. They do not establish performance on physical machinery or real IMU hardware.

## 2. System Architecture

```text
       Physics-Based Machine Simulator
                     |
              Synthetic 3-Axis IMU
                     |
           Signal Preprocessing
                     |
          DSP and Feature Extraction
                     |
              Anomaly Detection
                     |
            Probe and Diagnosis
                (ML + Rules)
                     |
             Agent Decision Plan
                     |
             Control Intervention
          /          |          \
    Harmonic      Speed       Derating /
    Rejection    Adjustment   Safe Stop
          \          |          /
                     |
           Independent Verification
                     |
             Recovery / Escalation
                     |
              Continuous Monitoring
```

The agent follows a state-driven workflow:

**MONITOR → ANOMALY → PROBE → DIAGNOSE → PLAN → ACT → VERIFY → RECOVER / SAFE STOP**

## 3. Main Capabilities

| Subsystem | Functionality |
|---|---|
| Machine simulation | Rotating-machine dynamics, structural response and configurable fault injection |
| Signal processing | Preprocessing, principal-axis fusion, FFT/order analysis, RMS, kurtosis, crest factor and envelope features |
| Anomaly detection | Healthy-baseline modelling and online abnormality detection |
| Fault diagnosis | Random Forest classification combined with physics-informed diagnostic rules |
| Active mitigation | Harmonic feed-forward FxLMS rejection and operating-speed interventions |
| Agent architecture | State-based monitoring, probing, diagnosis, planning, action and verification |
| Dashboard | Interactive Streamlit interface for scenario execution, experiments, visualisation and exports |

## 4. Fault Scenarios

The simulator supports seven operating classes:

1. Healthy operation
2. Imbalance
3. Misalignment
4. Resonance
5. Looseness
6. Bearing outer-race defect
7. Bearing inner-race defect

## 5. Repository Structure

```text
VIBRA-GUARD/
│
├── main.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── vibeagent/
│   ├── __init__.py
│   ├── agent.py
│   ├── anomaly.py
│   ├── config.py
│   ├── control.py
│   ├── diagnosis.py
│   ├── dsp.py
│   └── simulator.py
│
├── experiments/
│   ├── robust_validation.py
│   ├── robust_validation_v5_1.py
│   ├── bearing_analysis.py
│   ├── bearing_comparison.py
│   ├── fault_classifier.py
│   └── ... supporting experiments
│
├── results/
│   ├── v5_randomized_validation.csv
│   ├── v5_1_corrected_validation.csv
│   └── ... experimental figures
│
└── docs/
    ├── index.html
    ├── styles.css
    └── script.js
```

## 6. Installation

**Requirements:** Python 3.10 or later.

### Clone the repository

```bash
git clone https://github.com/suresh-vlsi/VIBRA-GUARD.git
cd VIBRA-GUARD
```

### Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / Ubuntu / WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Running the Application

### Command-line execution

Quick experiment:

```bash
python main.py --quick
```

Full experiment:

```bash
python main.py
```

### Interactive dashboard

```bash
streamlit run dashboard.py
```

Open the local URL provided by Streamlit, usually:

`http://localhost:8501`

## 8. Experimental Validation

The repository contains multiple generations of simulation experiments. Their results must be interpreted separately because they use different evaluation runs and configurations.

### V4.0.0 — Main application quick validation

The V4 quick validation reported:

| Metric | Result |
|---|---:|
| Random Forest accuracy | 97.9% |
| Anomaly detection F1 | 0.987 |
| Healthy false-alarm rate | 0.0% |
| Fault detection recall | 97.5% |
| Fused diagnosis accuracy | 100.0% |

The fused diagnosis result covers the six fault classes only (120 samples). It is not an overall seven-class healthy-plus-fault accuracy.

**Closed-loop results**

| Fault | Vibration reduction | Outcome |
|---|---:|---|
| Imbalance | 13.88 dB | Recovered |
| Misalignment | 12.27 dB | Mitigated, not recovered |
| Resonance | 21.89 dB | Recovered |
| Looseness | 4.76 dB | Mitigated, not recovered |
| Bearing outer | 4.37 dB | Mitigated, not recovered |
| Bearing inner | 7.54 dB | Mitigated, not recovered |

Two scenarios achieved verified recovery. Four achieved mitigation without verified recovery. No safety shutdown occurred in this quick run.

These results are from a specific simulation run, not hardware validation.

### V5 — Randomized simulation validation

The original V5 randomized experiment evaluated 140 trials: 20 trials per operating class, comprising 120 fault trials and 20 healthy trials.

| Metric | Result |
|---|---:|
| Fault detection | 85.8% |
| Correct fault diagnosis | 85.8% |
| Verified mitigation | 65.0% |
| Recovery | 30.8% |
| Safety stops | 27.5% |
| Healthy false alarms | 0% |

### V5.1 — Corrected randomized validation

The corrected V5.1 evaluation also contains 140 trials.

| Metric | Result |
|---|---:|
| Fault detection | 85.8% |
| Correct fault diagnosis | 84.2% |
| Verified mitigation | 64.2% |
| Recovery | 30.0% |
| Safety stops | 28.3% |
| Healthy false alarms | 0% |

The V5.1 dataset includes additional diagnostic confidence, severity, incident RMS and safety-trip information.

**Important distinction:** V5 and V5.1 are separate experimental evaluations. V5.1 is not a new physical hardware release, and its corrected evaluation does not automatically imply a change to the main V4 agent implementation.

### Reproducing randomized experiments

```bash
python -m experiments.robust_validation --trials 5 --quiet

python -m experiments.robust_validation_v5_1 --trials 5 --quiet
```

For a larger V5.1 experiment:

```bash
python -m experiments.robust_validation_v5_1 --trials 20 --quiet
```

The scripts generate validation CSV files under `results/`. Always report trial counts, configuration and simulation limitations alongside performance percentages.

## 9. Technology Stack

- **Programming:** Python
- **Numerical computing:** NumPy, SciPy
- **Machine learning:** scikit-learn
- **Signal visualisation:** Matplotlib
- **Data handling:** Pandas
- **Interactive interface:** Streamlit
- **Control:** Harmonic feed-forward FxLMS adaptation

## 10. Limitations and Future Work

VIBRA-GUARD is currently a software-only simulation and research prototype.

Current limitations:

- No physical IMU, motor rig or actuator has been tested.
- Synthetic signals and simulator assumptions may not represent all real-world machine behaviour.
- The single-mode and secondary-path assumptions limit generalisation to complex multi-mode structures.
- Bearing-impact components are broadband and impulsive; harmonic FxLMS rejection is not expected to cancel all such energy.
- Safety thresholds and mitigation policies are simulation parameters, not certified machine-protection settings.

Potential future extensions:

1. Physical IMU integration and sensor calibration.
2. Hardware-in-the-loop experimentation.
3. Real-time embedded control implementation.
4. Validation using representative rotating-machine datasets.
5. Multi-mode structural modelling and secondary-path identification.
6. Independent safety engineering and controlled laboratory testing.

## 11. Project Status

**Current release:** V4.0.0

VIBRA-GUARD demonstrates an integrated simulation-based workflow for vibration monitoring, fault diagnosis, active rejection and independent mitigation verification.

The V5 and V5.1 experiments provide additional randomized evaluation of the monitoring and mitigation approach.

Physical sensor integration and hardware-based validation remain future development objectives.

---

## Online Resources

| Resource | Link |
|---|---|
| Project Portfolio | [GitHub Pages](https://suresh-vlsi.github.io/VIBRA-GUARD/) |
| Live Dashboard | [Streamlit](https://vibra-guard.streamlit.app/) |
| Source Code | [GitHub Repository](https://github.com/suresh-vlsi/VIBRA-GUARD) |
| Release | [VIBRA-GUARD V4.0.0](https://github.com/suresh-vlsi/VIBRA-GUARD/releases/tag/v4.0.0) |

---

*Developed as a simulation-based engineering project exploring intelligent condition monitoring, machine learning, digital signal processing and active vibration control.*