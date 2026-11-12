import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # allow running from experiments/


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from vibeagent.simulator import Machine
from vibeagent.config import FS, BPFO

# Generate healthy machine
healthy = Machine(seed=42)
healthy_signal = healthy.step(8192)[:, 0]

# Generate faulty machine
faulty = Machine(seed=42)
faulty.inject("bearing_outer", 0.8)
faulty_signal = faulty.step(8192)[:, 0]

# Remove sensor offset and gravity
healthy_signal -= np.mean(healthy_signal)
faulty_signal -= np.mean(faulty_signal)

# Bandpass filter: 450 to 900 Hz
sos = signal.butter(
    4,
    [450, 900],
    btype="bandpass",
    fs=FS,
    output="sos"
)

healthy_bp = signal.sosfiltfilt(sos, healthy_signal)
faulty_bp = signal.sosfiltfilt(sos, faulty_signal)

# Calculate Hilbert envelope
healthy_env = np.abs(signal.hilbert(healthy_bp))
faulty_env = np.abs(signal.hilbert(faulty_bp))

# Plot first 0.2 seconds
samples = int(0.2 * FS)
t = np.arange(samples) / FS

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, healthy_bp[:samples])
plt.title("Healthy: High-frequency vibration")
plt.ylabel("Acceleration (g)")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, faulty_bp[:samples])
plt.title("Bearing fault: High-frequency vibration")
plt.ylabel("Acceleration (g)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, faulty_env[:samples])
plt.title("Bearing fault: Hilbert envelope")
plt.xlabel("Time (seconds)")
plt.ylabel("Envelope")
plt.grid()

plt.tight_layout()
plt.show()

print("Expected BPFO:", BPFO * 1200 / 60, "Hz")

# Envelope spectrum analysis

from scipy.signal import find_peaks

# Remove DC component from the envelope
env = faulty_env - np.mean(faulty_env)

N = len(env)

# Apply Hann window
window = np.hanning(N)

# Calculate FFT
spectrum = np.abs(np.fft.rfft(env * window))
frequency = np.fft.rfftfreq(N, d=1/FS)

# Normalize the single-sided amplitude spectrum
amplitude = 2 * spectrum / np.sum(window)

# Expected bearing fault frequency
expected_bpfo = BPFO * faulty.rpm / 60

# Find spectral peaks in the 10-300 Hz range
mask = (frequency >= 10) & (frequency <= 300)

peaks, _ = find_peaks(amplitude[mask])

peak_freqs = frequency[mask][peaks]
peak_amps = amplitude[mask][peaks]

# Print five strongest peaks
indices = np.argsort(peak_amps)[-5:][::-1]

print("\nFive strongest envelope peaks:")
for i in indices:
    print(f"{peak_freqs[i]:.2f} Hz : {peak_amps[i]:.5f} g")

print(f"\nExpected BPFO: {expected_bpfo:.2f} Hz")
print(f"FFT resolution: {FS/N:.4f} Hz")

# Plot envelope spectrum
plt.figure(figsize=(12, 5))

plt.plot(frequency, amplitude, label="Envelope FFT")

plt.axvline(
    expected_bpfo,
    color="red",
    linestyle="--",
    label=f"Expected BPFO = {expected_bpfo:.1f} Hz"
)

plt.xlim(0, 300)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Envelope amplitude (g)")
plt.title("Bearing Outer-Race Fault: Envelope Spectrum")
plt.legend()
plt.grid()

plt.show()
