import csv
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from data_pipeline import test_dataset, test_loader, PRETRAINED

if not PRETRAINED:
    raise ValueError(
        "ResNet needs PRETRAINED = True (224 x 224 images, ImageNet normalisation). "
        "Change it at the top of src/data_pipeline.py and run again."
    )


# The trained model to use. It must match a RUN_NAME from train_resnet.py.
RUN_NAME = "resnet18_subset"
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_FILE = RESULTS_DIR / f"{RUN_NAME}_best_seed{SEED}.pt"
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"
DETAILS_FILE = RESULTS_DIR / f"{RUN_NAME}_test_predictions.csv"

device = torch.device("cpu")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_FILE}\n"
        "Run src/train_resnet.py first, or check RUN_NAME and SEED above."
    )


model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"Loaded model: {MODEL_FILE.name}")
print(
    f"  Validation accuracy: {checkpoint['validation_accuracy']:.2%} "
    f"(epoch {checkpoint['epoch']}, stage {checkpoint['stage']})"
)
print(f"Test images: {len(test_dataset)}")


predictions = []

with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device)
        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        probability_dog = probabilities[:, 1]
        labels = outputs.argmax(dim=1)  # 0 = cat, 1 = dog

        for image_id, label, p_dog in zip(image_ids, labels, probability_dog):
            p_dog = p_dog.item()
            label = label.item()
            confidence = p_dog if label == 1 else 1 - p_dog
            predictions.append([int(image_id), label, p_dog, confidence])

predictions.sort(key=lambda row: row[0])

ids = [row[0] for row in predictions]
labels = [row[1] for row in predictions]

assert len(predictions) == 500, f"Expected 500 predictions, got {len(predictions)}"
assert len(set(ids)) == len(ids), "Some image IDs appear more than once"
assert set(labels) <= {0, 1}, "Labels must be 0 (cat) or 1 (dog)"

with open(SUBMISSION_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for image_id, label, _, _ in predictions:
        writer.writerow([image_id, label])

with open(DETAILS_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label", "predicted_class", "probability_dog", "confidence"])
    for image_id, label, p_dog, confidence in predictions:
        writer.writerow([
            image_id, label, "dog" if label == 1 else "cat",
            round(p_dog, 4), round(confidence, 4),
        ])

dog_count = sum(labels)
print(f"\nPredicted dogs: {dog_count} | Predicted cats: {len(labels) - dog_count}")

print("\nLeast confident predictions (good candidates to inspect for part e):")
for image_id, label, p_dog, confidence in sorted(predictions, key=lambda row: row[3])[:10]:
    print(
        f"  id {image_id:>3}: predicted {'dog' if label == 1 else 'cat'} "
        f"| P(dog) = {p_dog:.3f} | confidence {confidence:.1%}"
    )

print(f"\nSubmission saved to: {SUBMISSION_FILE}")
print(f"Detailed predictions saved to: {DETAILS_FILE}")