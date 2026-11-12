
import numpy as np

# Healthy baseline statistics

mu = np.array([0.06, 3.0])

cov = np.array([
    [0.0004, 0.0],
    [0.0, 0.25]
])

# Current measurement
healthy_sample = np.array([0.065, 3.1])

faulty_sample = np.array([0.30, 5.0])

def mahalanobis(x, mu, cov):

    diff = x - mu

    distance = np.sqrt(
        diff @ np.linalg.solve(cov, diff)
    )

    return distance

d1 = mahalanobis(healthy_sample, mu, cov)
d2 = mahalanobis(faulty_sample, mu, cov)

print("Healthy distance:", d1)
print("Faulty distance:", d2)

print("\nComparison:")

if d1 < 3:
    print("Healthy sample: Normal")

if d2 > 3:
    print("Faulty sample: Anomaly detected")
