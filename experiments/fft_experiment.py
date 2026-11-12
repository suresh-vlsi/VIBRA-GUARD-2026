
import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1000
duration = 1

t = np.arange(0, duration, 1/fs)

# Three vibration components
x = (
    0.5 * np.sin(2*np.pi*20*t)
    + 0.3 * np.sin(2*np.pi*40*t)
    + 0.2 * np.sin(2*np.pi*60*t)
)

# FFT
X = np.fft.rfft(x)
freq = np.fft.rfftfreq(len(x), 1/fs)

# Amplitude spectrum
amp = 2 * np.abs(X) / len(x)
amp[0] /= 2

# Find dominant frequency peaks
from scipy.signal import find_peaks

peaks, _ = find_peaks(amp, height=0.05)

print("Detected frequency components:")

for p in peaks:
    print(
        f"Frequency: {freq[p]:.1f} Hz, "
        f"Amplitude: {amp[p]:.3f} g"
    )

# Plot
plt.figure(figsize=(12, 5))
plt.plot(freq, amp)

plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (g)")
plt.title("FFT: Multiple Vibration Components")
plt.grid(True)

plt.show()
