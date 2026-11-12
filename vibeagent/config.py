"""Shared constants."""
FS = 2000            # IMU sample rate (Hz)
WIN = 1024           # analysis window (samples) ~0.51 s
RPM_NOM = 1200.0
RPM_MIN, RPM_MAX = 600.0, 1500.0   # allowed operating range of the process
FN_HEALTHY = 70.0    # structural natural frequency of the healthy machine (Hz)
ZETA = 0.03
BPFO, BPFI = 3.58, 5.42            # bearing fault orders (x shaft speed)
FAULTS = ["healthy", "imbalance", "misalignment", "resonance",
          "looseness", "bearing_outer", "bearing_inner"]
RMS_TRIP = 1.0       # g, hard safety limit -> immediate safe stop
ACT_LIMIT = 3.0      # actuator saturation (force units)
