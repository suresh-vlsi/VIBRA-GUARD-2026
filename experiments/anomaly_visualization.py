
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate healthy operating data
healthy = np.random.multivariate_normal(
    mean=[0.06, 3.0],
    cov=[
        [0.0004, 0.002],
        [0.002, 0.25]
    ],
    size=300
)

# Generate faulty observations
faulty = np.random.multivariate_normal(
    mean=[0.30, 5.0],
    cov=[
        [0.0004, 0.002],
        [0.002, 0.25]
    ],
    size=50
)

# Calculate healthy statistics
mu = np.mean(healthy, axis=0)
cov = np.cov(healthy.T)

# Calculate Mahalanobis distances
def calculate_distance(x):
    diff = x - mu

    return np.sqrt(
        diff @ np.linalg.solve(cov, diff)
    )

healthy_distances = np.array([
    calculate_distance(x) for x in healthy
])

faulty_distances = np.array([
    calculate_distance(x) for x in faulty
])

# Plot feature distribution
plt.figure(figsize=(10, 6))

plt.scatter(
    healthy[:, 0],
    healthy[:, 1],
    label="Healthy",
    alpha=0.6
)

plt.scatter(
    faulty[:, 0],
    faulty[:, 1],
    label="Faulty",
    alpha=0.7
)

plt.scatter(
    mu[0],
    mu[1],
    marker="x",
    s=150,
    label="Healthy Mean"
)

plt.xlabel("RMS (g)")
plt.ylabel("Kurtosis")
plt.title("Healthy vs Faulty Operating Conditions")
plt.legend()
plt.grid()
plt.show()

print("Healthy mean:", mu)
print("Healthy covariance:\n", cov)

print("\nAverage healthy distance:",
      np.mean(healthy_distances))

print("Average faulty distance:",
      np.mean(faulty_distances))
