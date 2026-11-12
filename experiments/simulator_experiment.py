import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # allow running from experiments/

from vibeagent.simulator import Machine
import matplotlib.pyplot as plt

machine = Machine()

machine.inject("bearing_outer", 0.8)

raw = machine.step(1024)

plt.figure(figsize=(12, 5))

plt.plot(raw[:, 0])

plt.title("Simulated Bearing Outer-Race Vibration")
plt.xlabel("Sample")
plt.ylabel("Acceleration (g)")

plt.grid()
plt.show()