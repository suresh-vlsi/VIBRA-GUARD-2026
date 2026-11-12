# VIBRA-GUARD fixes (drop-in, built on YOUR agent.py)

Only 6 files changed. Copy them into your repo at the same paths. Your calibration
settle step (`self.m.step(FS)`) in `agent.py` is kept. simulator.py, control.py,
config.py, anomaly.py, diagnosis.py are untouched.

| File | Change |
|---|---|
| vibeagent/agent.py | fixes 1-5 below |
| vibeagent/dsp.py | `vec()` compares keys by value, not identity |
| main.py | `run_scenario(..., rpm=None)`; resonance mode = 2 x shaft freq; detection/diagnosis only counted AFTER injection |
| dashboard.py | passes `rpm` properly (no global hack); NaN-safe metrics; `width="stretch"` |
| experiments/robust_validation_v5_1.py | pre-injection alarms no longer count as detections; hard-limit TRIP counts as a (labelled) detection |
| requirements.txt | minimum versions; `streamlit>=1.50` |

## Bugs fixed
1. After a failed ACTIVE_REJECT the next action was judged against the controller-assisted
   RMS (0.288 g), not the original incident RMS (0.329 g). Misalignment now reports 13.42 dB.
2. A failed ACTIVE_REJECT controller kept running when the agent escalated (DEGRADED path).
3. Speed planning used a fixed 1200 rpm; it now uses the machine's starting speed.
4. `recovery_rms_limit` was computed once; now a property that follows the current baseline.
5. `rms_before`/`wait` initialised; trip before diagnosis no longer says "suspected None".
6. Resonance scenario hard-coded 40 Hz (only resonant at 1200 rpm).
7. Validation metrics: pre-injection false alarms counted as detections; trip-caught faults
   counted as undetected.
8. Dashboard crash (`base=`/`diag=`), NaN display, deprecated `use_container_width`.

## Verified on your code
- Your unmodified code reproduces your log to the last digit (misalignment 0.2880 / 12.27 dB).
- With the fixes, only misalignment changes: 0.3290 / 13.42 dB. All other rows identical.
- Resonance at 900 rpm now diagnosed and recovered (17.3 dB); 1500 rpm trips the 1 g hard limit.

## README lines to update
- Misalignment: 12.27 dB -> 13.42 dB.
- V5.1 table: regenerate with `python -m experiments.robust_validation_v5_1 --trials 20 --quiet`
  (my run: detection 120/120 incl. 13 by hard-limit trip, diagnosis 86.7%, verified 64.2%,
  recovery 32.5%, safety stops 27.5%, healthy false alarms 0/20). Commit YOUR regenerated CSV.
- Delete the duplicate root-level robust_validation_v5_1.py if it exists.
