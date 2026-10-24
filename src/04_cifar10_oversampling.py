
# IE4483 - CIFAR-10 Oversampling
# Question (h): Solution 2

import numpy as np

# --------------------------------------------------
# 1. Set random seed
# --------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# --------------------------------------------------
# 2. Load imbalanced training dataset
# --------------------------------------------------
data = np.load("cifar10_imbalanced.npz")

X_train = data["X_train"]
y_train = data["y_train"]

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 3. Find target number of samples
# --------------------------------------------------
counts = np.bincount(y_train, minlength=10)
target_count = counts.max()

print("Target samples per class:", target_count)

# --------------------------------------------------
# 4. Oversample minority classes
# --------------------------------------------------
selected_indices = []

for class_id in range(10):

    class_indices = np.where(y_train == class_id)[0]

    if len(class_indices) < target_count:

        extra_indices = rng.choice(
            class_indices,
            size=target_count - len(class_indices),
            replace=True
        )

        class_indices = np.concatenate(
            [class_indices, extra_indices]
        )

    selected_indices.extend(class_indices)

selected_indices = np.array(selected_indices)
rng.shuffle(selected_indices)

X_train_oversampled = X_train[selected_indices]
y_train_oversampled = y_train[selected_indices]

# --------------------------------------------------
# 5. Verify class distribution
# --------------------------------------------------
new_counts = np.bincount(
    y_train_oversampled, minlength=10
)

print("\nClass distribution after oversampling:")

for i, name in enumerate(class_names):
    print(f"{name:12s}: {new_counts[i]}")

print("\nOriginal imbalanced shape:", X_train.shape)
print("Oversampled shape:        ", X_train_oversampled.shape)

# --------------------------------------------------
# 6. Save oversampled training data
# --------------------------------------------------
np.savez_compressed(
    "cifar10_oversampled.npz",
    X_train=X_train_oversampled,
    y_train=y_train_oversampled
)

print("\nOversampled dataset saved successfully!")
