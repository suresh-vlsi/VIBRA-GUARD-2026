
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

np.random.seed(42)

# Generate synthetic observations
healthy = np.random.multivariate_normal(
    [0.06, 3.0],
    [[0.0004, 0.002], [0.002, 0.25]],
    200
)

imbalance = np.random.multivariate_normal(
    [0.30, 2.8],
    [[0.0009, 0.003], [0.003, 0.30]],
    100
)

bearing = np.random.multivariate_normal(
    [0.16, 6.0],
    [[0.0005, 0.002], [0.002, 0.5]],
    100
)

# Combine features
X = np.vstack([healthy, imbalance, bearing])

# Assign labels
y = np.array(
    [0]*200 + [1]*100 + [2]*100
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Healthy",
        "Imbalance",
        "Bearing Outer"
    ]
))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=[
        "Healthy",
        "Imbalance",
        "Bearing Outer"
    ]
)

plt.title("Motor Fault Classification")
plt.show()
