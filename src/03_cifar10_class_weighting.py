
# IE4483 - CIFAR-10 Class Weighting
# Question (h): Solution 1

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# 1. Load the imbalanced dataset
# --------------------------------------------------
data = np.load("cifar10_imbalanced.npz")

y_train = data["y_train"]

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 2. Check class distribution
# --------------------------------------------------
classes = np.unique(y_train)

counts = np.bincount(y_train, minlength=10)

print("Training class distribution:")

for i, name in enumerate(class_names):
    print(f"{name:12s}: {counts[i]}")

# --------------------------------------------------
# 3. Calculate balanced class weights
# --------------------------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

# Convert weights to a dictionary for Keras
class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

# --------------------------------------------------
# 4. Display class weights
# --------------------------------------------------
print("\nCalculated class weights:")

for i, name in enumerate(class_names):
    print(f"{name:12s}: {class_weights[i]:.2f}")

print("\nClass weight dictionary:")
print(class_weights)

# --------------------------------------------------
# 5. Save class weights
# --------------------------------------------------
np.save("cifar10_class_weights.npy", class_weights)

print("\nClass weights saved successfully!")
