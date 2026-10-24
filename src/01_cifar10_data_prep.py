
# ============================================================
# IE4483 - CIFAR-10 Dataset Preparation
# Member 3 - Question (g)
# ============================================================

import setuptools  # Import first to fix distutils compatibility
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# 1. Load CIFAR-10
# ------------------------------------------------------------

(X_train_full, y_train_full), (X_test, y_test) = \
    tf.keras.datasets.cifar10.load_data()

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Original dataset:")
print("Training:", X_train_full.shape, y_train_full.shape)
print("Testing: ", X_test.shape, y_test.shape)

# ------------------------------------------------------------
# 2. Split training and validation sets
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

print("\nAfter splitting:")
print("Training:  ", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:   ", X_test.shape)

# ------------------------------------------------------------
# 3. Normalise pixel values
# ------------------------------------------------------------

X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nPixel range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())

# ------------------------------------------------------------
# 4. Check class distribution
# ------------------------------------------------------------

print("\nClass distribution:")

for i, name in enumerate(class_names):
    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)

    print(
        f"{name:12s} "
        f"Train={train_count:4d} "
        f"Val={val_count:4d} "
        f"Test={test_count:4d}"
    )

# ------------------------------------------------------------
# 5. Visualise one image per class
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]

    ax.imshow(X_train[idx])
    ax.set_title(class_names[i])
    ax.axis("off")

plt.suptitle("CIFAR-10 Sample Images")
plt.tight_layout()
plt.show()
