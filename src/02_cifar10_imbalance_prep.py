
# IE4483 - CIFAR-10 Class Imbalance Preparation

import setuptools  # Import first to fix distutils compatibility
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Set random seed
# --------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# --------------------------------------------------
# 2. Load CIFAR-10
# --------------------------------------------------
(X_train_full, y_train_full), (X_test, y_test) = \
    tf.keras.datasets.cifar10.load_data()

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 3. Split training and validation data
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

# --------------------------------------------------
# 4. Normalise images
# --------------------------------------------------
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# --------------------------------------------------
# 5. Create an imbalanced training dataset
# --------------------------------------------------
# CIFAR-10 labels: cat = 3, dog = 5
minority_classes = [3, 5]
minority_count = 400

selected_indices = []

for class_id in range(10):

    class_indices = np.where(y_train == class_id)[0]

    if class_id in minority_classes:
        class_indices = rng.choice(
            class_indices,
            size=minority_count,
            replace=False
        )

    selected_indices.extend(class_indices)

selected_indices = np.array(selected_indices)
rng.shuffle(selected_indices)

X_train_imbalanced = X_train[selected_indices]
y_train_imbalanced = y_train[selected_indices]

# --------------------------------------------------
# 6. Check class distributions
# --------------------------------------------------
balanced_counts = np.bincount(y_train, minlength=10)
imbalanced_counts = np.bincount(
    y_train_imbalanced, minlength=10
)

print("Class distribution:")
print(f"{'Class':12s} {'Balanced':>10s} {'Imbalanced':>12s}")

for i, name in enumerate(class_names):
    print(
        f"{name:12s} "
        f"{balanced_counts[i]:10d} "
        f"{imbalanced_counts[i]:12d}"
    )

print("\nDataset shapes:")
print("Balanced train:  ", X_train.shape)
print("Imbalanced train:", X_train_imbalanced.shape)
print("Validation:      ", X_val.shape)
print("Test:            ", X_test.shape)

# --------------------------------------------------
# 7. Plot class distribution
# --------------------------------------------------
x = np.arange(10)
width = 0.35

plt.figure(figsize=(12, 5))

plt.bar(
    x - width / 2,
    balanced_counts,
    width,
    label="Balanced"
)

plt.bar(
    x + width / 2,
    imbalanced_counts,
    width,
    label="Imbalanced"
)

plt.xticks(x, class_names, rotation=45)
plt.ylabel("Number of training images")
plt.title("CIFAR-10: Balanced vs Imbalanced Training Data")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Save dataset for later experiments
# --------------------------------------------------
np.savez_compressed(
    "cifar10_imbalanced.npz",
    X_train=X_train_imbalanced,
    y_train=y_train_imbalanced,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\nImbalanced dataset saved successfully!")
