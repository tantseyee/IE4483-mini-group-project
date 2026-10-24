
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_DIR = BASE_DIR / "datasets" / "val"

CHECKPOINT_PATH = (
    BASE_DIR / "checkpoints"
    / "resnet18_subset_best_seed123.pt"
)

OUTPUT_DIR = BASE_DIR / "results" / "dogs_cats_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
CLASS_NAMES = ["cat", "dog"]

device = torch.device("cpu")
print("Using device:", device)

# --------------------------------------------------
# 2. Validation preprocessing
# --------------------------------------------------

validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

validation_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=validation_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(validation_dataset))
print("Class mapping:", validation_dataset.class_to_idx)

# --------------------------------------------------
# 3. Load ResNet-18 checkpoint
# --------------------------------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Checkpoint seed:", checkpoint["seed"])
print("Saved validation accuracy:",
      f"{checkpoint['validation_accuracy']:.2%}")

# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

all_true = []
all_pred = []
all_confidence = []

total_batches = len(validation_loader)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(validation_loader, start=1):
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        all_true.extend(labels.numpy())
        all_pred.extend(predictions.cpu().numpy())
        all_confidence.extend(confidence.cpu().numpy())

        # Display progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"Processing batch {batch_idx}/{total_batches} "
                f"({batch_idx / total_batches * 100:.1f}%)",
                flush=True
            )

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_confidence = np.array(all_confidence)

accuracy = accuracy_score(all_true, all_pred)

print("\nValidation accuracy:", f"{accuracy:.2%}")

print("\nClassification report:")
report = classification_report(
    all_true,
    all_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt", "w"
) as f:
    f.write(report)

# --------------------------------------------------
# 5. Save prediction results
# --------------------------------------------------

df = pd.DataFrame({
    "image_path": [
        path for path, _ in validation_dataset.samples
    ],
    "true_label": [
        CLASS_NAMES[int(x)] for x in all_true
    ],
    "predicted_label": [
        CLASS_NAMES[int(x)] for x in all_pred
    ],
    "confidence": all_confidence,
    "correct": all_true == all_pred
})

df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

print("\nCorrect predictions:", df["correct"].sum())
print("Incorrect predictions:", (~df["correct"]).sum())

# --------------------------------------------------
# 6. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(cmap="Blues", values_format="d")
plt.title("ResNet-18 Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "validation_confusion_matrix.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 7. Show correct and incorrect examples
# --------------------------------------------------

correct_examples = df[df["correct"]].head(3)
incorrect_examples = df[~df["correct"]].head(3)

selected = pd.concat([
    correct_examples,
    incorrect_examples
])

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (_, sample) in zip(axes.flat, selected.iterrows()):
    with Image.open(sample["image_path"]) as img:
        ax.imshow(img.convert("RGB"))

    ax.set_title(
        f"True: {sample['true_label']}\n"
        f"Predicted: {sample['predicted_label']}\n"
        f"Confidence: {sample['confidence']:.2%}",
        fontsize=10
    )
    ax.axis("off")

# Hide unused panels if fewer than 6 samples
for ax in axes.flat[len(selected):]:
    ax.axis("off")

plt.suptitle(
    "ResNet-18: Correct and Incorrect Validation Predictions",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "validation_prediction_examples.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nResults saved to:", OUTPUT_DIR)
