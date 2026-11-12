import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # allow running from experiments/


from vibeagent.config import *

# Sampling calculations
Ts = 1 / FS
nyquist = FS / 2
resolution = FS / WIN
duration = WIN / FS

# Rotational frequency
fr = RPM_NOM / 60

# Bearing fault frequencies
bpfo = BPFO * fr
bpfi = BPFI * fr

# Harmonics
print("ROTATIONAL FREQUENCY ANALYSIS")
for k in [0.5, 1, 1.5, 2, 3, 4, 5]:
    print(f"{k}x = {k * fr:.3f} Hz")

print("\nSAMPLING PARAMETERS")
print("Sampling interval:", Ts, "seconds")
print("Nyquist frequency:", nyquist, "Hz")
print("FFT resolution:", resolution, "Hz")
print("Window duration:", duration, "seconds")

print("\nBEARING FREQUENCIES")
print("BPFO:", bpfo, "Hz")
print("BPFI:", bpfi, "Hz")

print("\nRESONANCE ANALYSIS")
resonance_rpm = FN_HEALTHY * 60 / 3
print("RPM for 3x resonance:", resonance_rpm)
