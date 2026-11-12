
import numpy as np

# Simulated healthy feature collection

features_list = []

for i in range(5):

    rms = np.random.uniform(0.05, 0.08)
    kurt = np.random.uniform(2.8, 3.3)
    crest = np.random.uniform(1.5, 2.0)

    feature = {
        "rms": rms,
        "kurt": kurt,
        "crest": crest
    }

    features_list.append(feature)

# Display results

for i, f in enumerate(features_list):

    print("Sample:", i + 1)
    print("RMS:", round(f["rms"], 3))
    print("Kurtosis:", round(f["kurt"], 3))
    print("Crest factor:", round(f["crest"], 3))
    print("----------------------")
