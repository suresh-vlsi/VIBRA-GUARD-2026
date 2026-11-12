import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # allow running from experiments/


import numpy as np
from scipy import signal

from vibeagent.simulator import Machine
from vibeagent.config import FS, BPFO

# Generate signals
healthy = Machine(seed=42)
faulty = Machine(seed=42)

faulty.inject("bearing_outer", 0.8)

N = 32768

x_h = healthy.step(N)[:, 0]
x_f = faulty.step(N)[:, 0]

# Remove DC
x_h -= np.mean(x_h)
x_f -= np.mean(x_f)

# Bandpass filtering
sos = signal.butter(
    4, [450, 900],
    btype="bandpass",
    fs=FS,
    output="sos"
)

h = signal.sosfiltfilt(sos, x_h)
f = signal.sosfiltfilt(sos, x_f)

# Hilbert envelope
env_h = np.abs(signal.hilbert(h))
env_f = np.abs(signal.hilbert(f))

# Envelope spectrum
def get_spectrum(env):
    env = env - np.mean(env)
    window = np.hanning(len(env))

    A = np.abs(np.fft.rfft(env * window))
    A = 2 * A / np.sum(window)

    freq = np.fft.rfftfreq(len(env), 1/FS)

    return freq, A

freq, Ah = get_spectrum(env_h)
_, Af = get_spectrum(env_f)

# Theoretical frequencies
rpm = faulty.rpm
bpfo = BPFO * rpm / 60

print("Expected BPFO:", bpfo)

# Find strongest peak around each harmonic
for k in range(1, 4):

    target = k * bpfo

    mask = (
        (freq >= target - 3) &
        (freq <= target + 3)
    )

    peak_h = np.max(Ah[mask])
    peak_f = np.max(Af[mask])

    ratio = peak_f / (peak_h + 1e-12)

    snr_db = 20 * np.log10(ratio)

    print(f"\nHarmonic {k}")
    print("Expected frequency:", target)
    print("Healthy amplitude:", peak_h)
    print("Fault amplitude:", peak_f)
    print("Fault/healthy ratio:", ratio)
    print("Relative level (dB):", snr_db)
