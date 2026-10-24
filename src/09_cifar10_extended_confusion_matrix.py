
# IE4483 - Extended CIFAR-10 Confusion Matrix Analysis
# Question (h): 10-epoch CNN experiments

import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------
BATCH_SIZE = 64

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

results_dir = Path("results/extended")

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

models = {
    "balanced": "balanced_model.pth",
    "imbalanced": "imbalanced_model.pth",
    "class_weighting": "class_weighting_model.pth",
    "oversampling": "oversampling_model.pth"
}

# --------------------------------------------------
# 2. CNN architecture (same as previous scripts)
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
# 3. Load unchanged CIFAR-10 test dataset
# --------------------------------------------------
with np.load("cifar10_imbalanced.npz") as data:
    X_test = data["X_test"]
    y_test = data["y_test"]

X_test = np.ascontiguousarray(
    np.transpose(X_test, (0, 3, 1, 2))
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test.astype(np.int64))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --------------------------------------------------
# 4. Evaluate each 10-epoch model
# --------------------------------------------------
for name, filename in models.items():

    print(f"\n=== {name.upper()} ===", flush=True)

    model = BaselineCNN().to(device)

    model.load_state_dict(
        torch.load(
            results_dir / filename,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    # --------------------------------------------------
    # 5. Classification report
    # --------------------------------------------------
    report = classification_report(
        actual,
        predicted,
        labels=np.arange(10),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(report, flush=True)

    report_file = (
        results_dir / f"{name}_classification_report.txt"
    )

    report_file.write_text(report)

    # --------------------------------------------------
    # 6. Confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        actual,
        predicted,
        labels=np.arange(10)
    )

    # Normalize each row to show proportions
    cm_normalized = cm.astype(float) / cm.sum(
        axis=1, keepdims=True
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        colorbar=True
    )

    ax.set_title(
        f"CIFAR-10 Confusion Matrix (10 Epochs) - {name}"
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    figure_file = (
        results_dir / f"{name}_confusion_matrix.png"
    )

    plt.savefig(figure_file, dpi=300)
    plt.close(fig)

    # Save raw confusion matrix
    np.savetxt(
        results_dir / f"{name}_confusion_counts.csv",
        cm,
        fmt="%d",
        delimiter=","
    )

    # --------------------------------------------------
    # 7. Analyse cat and dog predictions
    # --------------------------------------------------
    print("\nCat images predicted as:")

    for i in np.argsort(cm[3])[::-1][:5]:
        print(f"  {class_names[i]}: {cm[3, i]}")

    print("\nDog images predicted as:")

    for i in np.argsort(cm[5])[::-1][:5]:
        print(f"  {class_names[i]}: {cm[5, i]}")

    print("\nSaved report and confusion matrix.")

print("\nAll 10-epoch confusion matrices generated!")
