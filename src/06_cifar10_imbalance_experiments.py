
# IE4483 - CIFAR-10 Class Imbalance Experiments
# Question (h): Imbalanced baseline, class weighting,
# and oversampling

import gc
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device, flush=True)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 2. CNN architecture (same as Question g)
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
            nn.MaxPool2d(2),
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
# 3. Convert images to PyTorch format
# --------------------------------------------------
def make_loader(X, y, shuffle=False):
    # NumPy: (N, H, W, C)
    # PyTorch: (N, C, H, W)
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
# 4. Predict labels and calculate metrics
# --------------------------------------------------
def get_predictions(model, loader):
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
    recall = recall_score(
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
        "cat_recall": recall[3],
        "dog_recall": recall[5]
    }


# --------------------------------------------------
# 5. Load the unchanged validation/test datasets
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
# 6. Evaluate saved balanced baseline
# --------------------------------------------------
print("\n=== Balanced baseline ===", flush=True)

balanced_model = BaselineCNN().to(device)

balanced_model.load_state_dict(
    torch.load(
        results_dir / "cifar10_balanced_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

actual, predicted = get_predictions(
    balanced_model, test_loader
)

balanced_metrics = calculate_metrics(actual, predicted)

print(balanced_metrics, flush=True)

del balanced_model
gc.collect()

# --------------------------------------------------
# 7. Train one experimental model
# --------------------------------------------------
def train_experiment(name, dataset_file, use_weights=False):

    print(f"\n=== {name} ===", flush=True)

    # Reset seed for a fairer comparison
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with np.load(dataset_file) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]

    print("Training samples:", len(y_train), flush=True)

    train_loader = make_loader(
        X_train, y_train, shuffle=True
    )

    del X_train, y_train
    gc.collect()

    # Fresh CNN with the same initial seed
    model = BaselineCNN().to(device)

    # Default: ordinary cross-entropy loss
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

        loss_function = nn.CrossEntropyLoss(
            weight=weights
        )
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training loop
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

        val_actual, val_predicted = get_predictions(
            model, val_loader
        )

        val_accuracy = accuracy_score(
            val_actual, val_predicted
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Validation accuracy: {val_accuracy:.2%}",
            flush=True
        )

    # Evaluate using unchanged test dataset
    actual, predicted = get_predictions(
        model, test_loader
    )

    metrics = calculate_metrics(actual, predicted)

    print("\nTest results:", flush=True)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}", flush=True)

    # Save trained model
    torch.save(
        model.cpu().state_dict(),
        results_dir / f"cifar10_{name}.pth"
    )

    del model, train_loader
    gc.collect()

    return metrics


# --------------------------------------------------
# 8. Run three imbalance experiments
# --------------------------------------------------
results = {
    "balanced": balanced_metrics
}

results["imbalanced"] = train_experiment(
    "imbalanced",
    "cifar10_imbalanced.npz"
)

results["class_weighting"] = train_experiment(
    "class_weighting",
    "cifar10_imbalanced.npz",
    use_weights=True
)

results["oversampling"] = train_experiment(
    "oversampling",
    "cifar10_oversampled.npz"
)

# --------------------------------------------------
# 9. Save comparison results
# --------------------------------------------------
import csv

output_file = results_dir / "cifar10_imbalance_comparison.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "experiment",
        "accuracy",
        "macro_f1",
        "cat_recall",
        "dog_recall"
    ])

    for name, metrics in results.items():
        writer.writerow([
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["cat_recall"],
            metrics["dog_recall"]
        ])

print("\n=== Final comparison ===", flush=True)

for name, metrics in results.items():
    print(
        f"{name:16s} | "
        f"Accuracy: {metrics['accuracy']:.2%} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"Cat recall: {metrics['cat_recall']:.2%} | "
        f"Dog recall: {metrics['dog_recall']:.2%}",
        flush=True
    )

print("\nAll experiments completed!", flush=True)
print("Comparison saved to:", output_file)
