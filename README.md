# ⚙️ VIBRA-GUARD
### Agentic Vibration Monitoring, Fault Diagnosis & Active Rejection

<p align="center">
  <a href="https://vibra-guard-2026.streamlit.app/">
    <img src="https://img.shields.io/badge/Live-Interactive%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="https://github.com/suresh-vlsi/VIBRA-GUARD-2026">
    <img src="https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Simulation%20Prototype-success?style=for-the-badge" />
</p>

**VIBRA-GUARD** is an agentic AI-based vibration monitoring and active rejection demonstrator for rotating machinery. It combines physics-based simulation, three-axis IMU signal processing, machine learning, fault diagnosis, intelligent decision-making and closed-loop vibration mitigation.

**Explore the project**

- 🚀 [Launch Live Dashboard](https://vibra-guard-2026.streamlit.app/)
- 💻 [Source Code – GitHub](https://github.com/suresh-vlsi/VIBRA-GUARD-2026)
- 🌐 [Project Portfolio](https://suresh-vlsi.github.io/VIBRA-GUARD/)

---

## 1. Project Overview

Rotating machinery can experience abnormal vibration due to mechanical faults such as imbalance, misalignment, resonance, looseness and bearing defects.

VIBRA-GUARD explores an intelligent monitoring system that can:

1. Generate synthetic three-axis IMU vibration signals.
2. Process vibration measurements using digital signal processing.
3. Detect abnormal machine behaviour.
4. Diagnose potential mechanical faults.
5. Select an appropriate mitigation strategy.
6. Apply simulated active vibration rejection or operating-point interventions.
7. Verify the mitigation outcome using subsequent measurements.

> **Validation scope:** This is a software-only, simulation-based engineering prototype. Reported results do not establish performance on physical machinery, real IMU hardware or a physical actuator.

## 2. System Architecture

```text
       PHYSICS-BASED MACHINE SIMULATOR
                      |
              SYNTHETIC 3-AXIS IMU
                      |
              SIGNAL PREPROCESSING
                      |
           DSP & FEATURE EXTRACTION
                      |
              ANOMALY DETECTION
                      |
             FAULT PROBING & DIAGNOSIS
                  (ML + Rules)
                      |
              AGENT DECISION PLANNING
                      |
               CONTROL INTERVENTION
             /          |          \
       Harmonic      Speed       Derating /
       Rejection   Adjustment    Safe Stop
             \          |          /
                      |
           MEASUREMENT-BASED VERIFICATION
                      |
              RECOVERY / ESCALATION
                      |
               CONTINUOUS MONITORING
```

### Agent workflow

**MONITOR → ANOMALY → PROBE → DIAGNOSE → PLAN → ACT → VERIFY → RECOVER / SAFE STOP**

The agent uses a state-driven workflow to connect fault detection, diagnosis, mitigation and verification.

## 3. Main Features

| Subsystem | Function |
|---|---|
| Machine simulator | Rotating-machine dynamics and configurable fault injection |
| IMU simulation | Synthetic three-axis vibration signals |
| Signal processing | Filtering, FFT, RMS, kurtosis and vibration features |
| Anomaly detection | Healthy-baseline modelling and abnormality detection |
| Fault diagnosis | Machine learning and physics-informed diagnostic rules |
| Active mitigation | Harmonic feed-forward FxLMS rejection and operating-speed interventions |
| Agent architecture | Monitoring, probing, diagnosis, planning, action and verification |
| Dashboard | Interactive Streamlit interface for simulation and visualisation |

## 4. Fault Scenarios

The simulator supports six fault conditions and healthy operation:

1. Imbalance
2. Misalignment
3. Resonance
4. Looseness
5. Bearing outer-race defect
6. Bearing inner-race defect
7. Healthy operation

## 5. Six-Fault Simulation Validation

A six-scenario simulation validation was performed using the uploaded validation dataset.

| Fault | Diagnosed | RMS Before (g) | RMS After (g) | Reduction (dB) | Agent State |
|---|---|---:|---:|---:|---|
| Imbalance | Yes | 0.304 | 0.0615 | 13.88 | MONITOR |
| Misalignment | Yes | 0.329 | 0.0702 | 13.42 | MITIGATED |
| Resonance | Yes | 0.662 | 0.0532 | 21.89 | MONITOR |
| Looseness | Yes | 0.769 | 0.4446 | 4.76 | MITIGATED |
| Bearing outer | Yes | 0.166 | 0.1003 | 4.37 | MITIGATED |
| Bearing inner | Yes | 0.256 | 0.1075 | 7.54 | MITIGATED |

### Agent intervention decisions

| Fault | Simulated intervention |
|---|---|
| Imbalance | ACTIVE_REJECT |
| Misalignment | ACTIVE_REJECT → DERATE |
| Resonance | SHIFT_SPEED |
| Looseness | OPTIMIZE_SPEED |
| Bearing outer | OPTIMIZE_SPEED |
| Bearing inner | DERATE |

**Interpretation**

- All six scenarios were diagnosed correctly in this validation dataset.
- All six scenarios showed a reduction in RMS vibration after intervention.
- The largest reported reduction was for resonance at approximately 21.89 dB.
- The smallest reported reduction was for bearing outer-race fault at approximately 4.37 dB.
- Agent state and intervention labels are reported as recorded in the dataset.

**Important:** This dataset contains six scenario records. It is not a large-scale randomized statistical evaluation and should not be interpreted as a generalisation or hardware-performance guarantee. The recorded agent states are not, by themselves, proof of independently verified recovery.

## 6. Experimental Validation History

The repository also contains earlier simulation experiments. These are separate evaluation runs and should not be combined with the six-scenario results above.

### Earlier quick validation

The existing project validation reported:

| Metric | Result |
|---|---:|
| Random Forest accuracy | 97.9% |
| Anomaly detection F1 | 0.987 |
| Healthy false-alarm rate | 0.0% |
| Fault detection recall | 97.5% |
| Fused diagnosis accuracy | 100.0% |

The fused diagnosis result covers six fault classes and 120 samples. It is not an overall seven-class healthy-plus-fault accuracy.

### Earlier randomized validation

The previous V5 and V5.1 experiments evaluated 140 trials each, consisting of 120 fault trials and 20 healthy trials.

| Metric | V5 | V5.1 |
|---|---:|---:|
| Fault detection | 85.8% | 85.8% |
| Correct fault diagnosis | 85.8% | 84.2% |
| Verified mitigation | 65.0% | 64.2% |
| Recovery | 30.8% | 30.0% |
| Safety stops | 27.5% | 28.3% |
| Healthy false alarms | 0% | 0% |

These are separate historical simulation evaluations, not results from the six-row validation dataset.

## 7. Repository Structure

```text
VIBRA-GUARD-2026/
│
├── main.py
├── dashboard.py
├── requirements.txt
├── README.md
├── CHANGES.md
├── robust_validation_v5_1.py
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
│   ├── anomaly_visualization.py
│   ├── bearing_analysis.py
│   ├── bearing_comparison.py
│   ├── bearing_snr.py
│   ├── engineering_calculations.py
│   ├── fault_classifier.py
│   ├── feature_learning.py
│   ├── fft_experiment.py
│   ├── legacy_robust_validation_v5.py
│   ├── mahalanobis_experiment.py
│   ├── robust_fault_classifier.py
│   ├── robust_validation_v5_1.py
│   ├── simulator_experiment.py
│   └── three_class_visualization.py
│
└── results/
    ├── events_bearing_inner.jsonl
    ├── events_bearing_outer.jsonl
    ├── events_imbalance.jsonl
    ├── events_looseness.jsonl
    ├── events_misalignment.jsonl
    ├── events_resonance.jsonl
    ├── scenarios.png
    ├── spectrum_imbalance.png
    └── v5_1_corrected_validation.csv
```

## 8. Installation

**Requirements:** Python 3.10 or later.

### Clone the repository

```bash
git clone https://github.com/suresh-vlsi/VIBRA-GUARD-2026.git
cd VIBRA-GUARD-2026
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

## 9. Running the Application

### Command-line execution

Quick experiment:

```bash
python main.py --quick
```

Full experiment:

```bash
python main.py
```

### Launch the interactive dashboard

```bash
streamlit run dashboard.py
```

Open the local Streamlit URL:

```text
http://localhost:8501
```

### Online dashboard

The deployed application is available at:

**https://vibra-guard-2026.streamlit.app/**

## 10. Technology Stack

| Technology | Application |
|---|---|
| Python | Core implementation |
| NumPy | Numerical computing |
| SciPy | Signal processing |
| scikit-learn | Machine learning |
| Matplotlib | Data visualisation |
| Pandas | Experimental data handling |
| Streamlit | Interactive dashboard |
| FxLMS | Feed-forward active vibration rejection |

## 11. Limitations and Future Work

VIBRA-GUARD is currently a software-only simulation and research prototype.

### Current limitations

- No physical IMU, motor rig or actuator has been tested.
- Synthetic signals and simulator assumptions may not represent all real-world machine behaviour.
- Single-mode and secondary-path assumptions limit generalisation to complex structures.
- Harmonic FxLMS rejection is not expected to cancel all broadband bearing-impact energy.
- Safety thresholds and mitigation policies are simulation parameters, not certified machine-protection settings.

### Future development

1. Physical three-axis IMU integration and sensor calibration.
2. Hardware-in-the-loop experimentation.
3. Real-time embedded control implementation.
4. Validation using representative rotating-machine datasets.
5. Multi-mode structural modelling and secondary-path identification.
6. Independent safety engineering and controlled laboratory testing.

## 12. Project Status

**Project:** VIBRA-GUARD – Agentic Vibration Monitoring, Fault Diagnosis and Active Rejection

**Current implementation:** Python-based simulation and interactive dashboard.

**Demonstrated capabilities:** Synthetic vibration generation, anomaly detection, fault diagnosis, agentic intervention selection, simulated active rejection and measurement-based evaluation.

**Physical prototype:** Future development objective.

---

## Online Resources

| Resource | Link |
|---|---|
| Live Dashboard | [Streamlit Community Cloud](https://vibra-guard-2026.streamlit.app/) |
| Source Code | [VIBRA-GUARD-2026 on GitHub](https://github.com/suresh-vlsi/VIBRA-GUARD-2026) |
| Project Portfolio | [GitHub Pages](https://suresh-vlsi.github.io/VIBRA-GUARD/) |

---

*Developed as an engineering project exploring agentic AI, intelligent condition monitoring, machine learning, digital signal processing and active vibration control.*