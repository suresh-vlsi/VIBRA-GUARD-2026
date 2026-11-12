
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# More overlapping synthetic distributions
healthy = np.random.multivariate_normal(
    [0.08, 3.0],
    [[0.0016, 0.005], [0.005, 0.8]],
    500
)

imbalance = np.random.multivariate_normal(
    [0.13, 3.4],
    [[0.0025, 0.006], [0.006, 1.0]],
    500
)

bearing = np.random.multivariate_normal(
    [0.11, 4.5],
    [[0.0020, 0.004], [0.004, 1.2]],
    500
)

# Combine observations
X = np.vstack([healthy, imbalance, bearing])

y = np.array(
    [0]*500 + [1]*500 + [2]*500
)

# Add measurement noise
X += np.random.normal(0, [0.008, 0.2], X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=[
        "Healthy",
        "Imbalance",
        "Bearing"
    ]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
