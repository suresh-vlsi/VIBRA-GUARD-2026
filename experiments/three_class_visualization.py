
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Healthy motor
healthy = np.random.multivariate_normal(
    [0.06, 3.0],
    [[0.0004, 0.002], [0.002, 0.25]],
    200
)

# Imbalance
imbalance = np.random.multivariate_normal(
    [0.30, 2.8],
    [[0.0009, 0.003], [0.003, 0.30]],
    100
)

# Bearing outer-race fault
bearing = np.random.multivariate_normal(
    [0.16, 6.0],
    [[0.0005, 0.002], [0.002, 0.5]],
    100
)

plt.figure(figsize=(10, 6))

plt.scatter(healthy[:, 0], healthy[:, 1],
            label="Healthy", alpha=0.6)

plt.scatter(imbalance[:, 0], imbalance[:, 1],
            label="Imbalance", alpha=0.6)

plt.scatter(bearing[:, 0], bearing[:, 1],
            label="Bearing Outer Fault", alpha=0.6)

plt.xlabel("RMS (g)")
plt.ylabel("Kurtosis")

plt.title("Three-Class Motor Fault Analysis")

plt.legend()
plt.grid()
plt.show()
