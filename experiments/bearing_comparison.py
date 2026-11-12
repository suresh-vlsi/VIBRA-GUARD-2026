import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # allow running from experiments/


import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from vibeagent.simulator import Machine
from vibeagent.config import FS, BPFO

# Generate identical-seed healthy and faulty machines
healthy = Machine(seed=42)
faulty = Machine(seed=42)

faulty.inject("bearing_outer", 0.8)

N = 32768

x_healthy = healthy.step(N)[:, 0]
x_faulty = faulty.step(N)[:, 0]

# Remove DC
x_healthy -= np.mean(x_healthy)
x_faulty -= np.mean(x_faulty)

# High-frequency bandpass
sos = signal.butter(
    4, [450, 900],
    btype="bandpass",
    fs=FS,
    output="sos"
)

# Filter signals
h = signal.sosfiltfilt(sos, x_healthy)
f = signal.sosfiltfilt(sos, x_faulty)

# Extract envelopes
env_h = np.abs(signal.hilbert(h))
env_f = np.abs(signal.hilbert(f))

# Calculate envelope FFT
def envelope_fft(env):
    env = env - np.mean(env)
    window = np.hanning(len(env))

    spectrum = np.abs(
        np.fft.rfft(env * window)
    )

    freq = np.fft.rfftfreq(len(env), 1 / FS)
    amp = 2 * spectrum / np.sum(window)

    return freq, amp

freq, amp_h = envelope_fft(env_h)
_, amp_f = envelope_fft(env_f)

# Expected BPFO
rpm = faulty.rpm
bpfo = BPFO * rpm / 60

# Plot comparison
plt.figure(figsize=(12, 6))

plt.plot(freq, amp_h, label="Healthy")
plt.plot(freq, amp_f, label="Bearing outer fault")

plt.axvline(
    bpfo,
    color="red",
    linestyle="--",
    label=f"BPFO = {bpfo:.1f} Hz"
)

plt.xlim(0, 300)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Envelope amplitude (g)")
plt.title("Healthy vs Bearing Fault: Envelope FFT")

plt.legend()
plt.grid()
plt.show()

print("Expected BPFO:", bpfo, "Hz")
