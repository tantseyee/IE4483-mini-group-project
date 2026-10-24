
# IE4483 - Extended CIFAR-10 Experiments
# Question (h): 10-epoch comparison

import csv
import gc
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------
SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device, flush=True)

results_dir = Path("results/extended")
results_dir.mkdir(parents=True, exist_ok=True)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 2. CNN architecture
# --------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, images):
        images = self.features(images)
        return self.classifier(images)


# --------------------------------------------------
# 3. Prepare PyTorch DataLoader
# --------------------------------------------------
def make_loader(X, y, shuffle=False):

    X = np.ascontiguousarray(
        np.transpose(X, (0, 3, 1, 2))
    )

    images = torch.from_numpy(X)
    labels = torch.from_numpy(
        y.astype(np.int64)
    )

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


# --------------------------------------------------
# 4. Evaluation
# --------------------------------------------------
def predict(model, loader):
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    return np.array(actual), np.array(predicted)


def calculate_metrics(actual, predicted):

    recalls = recall_score(
        actual,
        predicted,
        labels=np.arange(10),
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=np.arange(10),
            average="macro",
            zero_division=0
        ),
        "cat_recall": recalls[3],
        "dog_recall": recalls[5]
    }


# --------------------------------------------------
# 5. Load unchanged validation and test sets
# --------------------------------------------------
with np.load("cifar10_imbalanced.npz") as data:
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

del X_val, X_test
gc.collect()

# --------------------------------------------------
# 6. Load balanced CIFAR-10 training data
# --------------------------------------------------
# Our earlier TensorFlow setup requires setuptools first
import setuptools
import tensorflow as tf

(X_full, y_full), _ = tf.keras.datasets.cifar10.load_data()

y_full = y_full.flatten()

X_balanced, _, y_balanced, _ = train_test_split(
    X_full,
    y_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_full
)

X_balanced = X_balanced.astype("float32") / 255.0

print("Balanced training samples:", len(y_balanced))

# --------------------------------------------------
# 7. Save identical initial CNN weights
# --------------------------------------------------
initial_model = BaselineCNN()

initial_weights = {
    key: value.clone()
    for key, value in initial_model.state_dict().items()
}

del initial_model

# --------------------------------------------------
# 8. Define training experiment
# --------------------------------------------------
def train_experiment(name, X_train, y_train, use_weights=False):

    print(f"\n=== {name.upper()} ===", flush=True)

    # Reset random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader = make_loader(
        X_train, y_train, shuffle=True
    )

    model = BaselineCNN().to(device)

    # Load identical initial weights
    model.load_state_dict(initial_weights)

    if use_weights:
        weights_dict = np.load(
            "cifar10_class_weights.npy",
            allow_pickle=True
        ).item()

        weights = torch.tensor(
            [weights_dict[i] for i in range(10)],
            dtype=torch.float32,
            device=device
        )

        loss_function = nn.CrossEntropyLoss(weight=weights)

    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(EPOCHS):
        model.train()

        correct = 0
        total = 0
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = correct / total

        val_actual, val_predicted = predict(
            model, val_loader
        )

        val_accuracy = accuracy_score(
            val_actual, val_predicted
        )

        history.append([
            epoch + 1,
            total_loss / len(train_loader),
            train_accuracy,
            val_accuracy
        ])

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train: {train_accuracy:.2%} | "
            f"Validation: {val_accuracy:.2%}",
            flush=True
        )

    # Evaluate on test set
    actual, predicted = predict(model, test_loader)

    metrics = calculate_metrics(actual, predicted)

    print("\nTest metrics:", metrics, flush=True)

    # Save model
    torch.save(
        model.cpu().state_dict(),
        results_dir / f"{name}_model.pth"
    )

    # Save training history
    np.savetxt(
        results_dir / f"{name}_history.csv",
        np.array(history),
        delimiter=",",
        header="epoch,loss,train_accuracy,val_accuracy",
        comments=""
    )

    del model, train_loader
    gc.collect()

    return metrics, np.array(history)


# --------------------------------------------------
# 9. Run all four experiments
# --------------------------------------------------
results = {}
histories = {}

# Experiment 1: Balanced baseline
results["balanced"], histories["balanced"] = train_experiment(
    "balanced",
    X_balanced,
    y_balanced
)

del X_balanced, y_balanced
gc.collect()

# Experiment 2: Imbalanced baseline
with np.load("cifar10_imbalanced.npz") as data:
    X_train = data["X_train"]
    y_train = data["y_train"]

results["imbalanced"], histories["imbalanced"] = train_experiment(
    "imbalanced",
    X_train,
    y_train
)

# Experiment 3: Class weighting
results["class_weighting"], histories["class_weighting"] = train_experiment(
    "class_weighting",
    X_train,
    y_train,
    use_weights=True
)

del X_train, y_train
gc.collect()

# Experiment 4: Oversampling
with np.load("cifar10_oversampled.npz") as data:
    X_over = data["X_train"]
    y_over = data["y_train"]

results["oversampling"], histories["oversampling"] = train_experiment(
    "oversampling",
    X_over,
    y_over
)

del X_over, y_over
gc.collect()

# --------------------------------------------------
# 10. Save comparison table
# --------------------------------------------------
with open(
    results_dir / "extended_comparison.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "experiment", "accuracy", "macro_f1",
        "cat_recall", "dog_recall"
    ])

    for name, metrics in results.items():
        writer.writerow([
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["cat_recall"],
            metrics["dog_recall"]
        ])

# --------------------------------------------------
# 11. Plot validation accuracy
# --------------------------------------------------
plt.figure(figsize=(9, 6))

for name, history in histories.items():
    plt.plot(
        history[:, 0],
        history[:, 3] * 100,
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("CIFAR-10: Validation Accuracy Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "validation_accuracy_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 12. Print final comparison
# --------------------------------------------------
print("\n=== FINAL 10-EPOCH COMPARISON ===")

for name, metrics in results.items():
    print(
        f"{name:16s} | "
        f"Accuracy: {metrics['accuracy']:.2%} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"Cat recall: {metrics['cat_recall']:.2%} | "
        f"Dog recall: {metrics['dog_recall']:.2%}"
    )

print("\nExtended experiments completed!")
print("Results saved to:", results_dir)
