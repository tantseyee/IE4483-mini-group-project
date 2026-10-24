
# IE4483 - CIFAR-10 Multiclass Classification
# Question (g): Adapt Member 1's baseline CNN

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use Mac GPU when available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# 2. Load balanced CIFAR-10 dataset
# --------------------------------------------------
# Import setuptools first for our TensorFlow setup
import setuptools
import tensorflow as tf

(X_train_full, y_train_full), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Same split as 01_cifar10_data_prep.py
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

# Normalize pixel values to 0-1
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))

# --------------------------------------------------
# 3. Convert NumPy arrays to PyTorch datasets
# --------------------------------------------------
def make_dataset(images, labels):
    # PyTorch CNN expects (N, C, H, W)
    # CIFAR-10 is originally (N, H, W, C)
    images = np.transpose(images, (0, 3, 1, 2))
    images = np.ascontiguousarray(images)

    image_tensor = torch.from_numpy(images)
    label_tensor = torch.from_numpy(
        labels.astype("int64")
    )

    return TensorDataset(image_tensor, label_tensor)


train_dataset = make_dataset(X_train, y_train)
val_dataset = make_dataset(X_val, y_val)
test_dataset = make_dataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 4. CNN architecture adapted from Member 1
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
            nn.Linear(64, 10),  # 10 CIFAR-10 classes
        )

    def forward(self, images):
        images = self.features(images)
        return self.classifier(images)


model = BaselineCNN().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# 5. Evaluation function
# --------------------------------------------------
def evaluate(data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# --------------------------------------------------
# 6. Train the CNN
# --------------------------------------------------
history = []

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

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

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    val_accuracy = evaluate(val_loader)
    average_loss = total_loss / len(train_loader)

    history.append([
        epoch + 1,
        average_loss,
        train_accuracy,
        val_accuracy
    ])

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation accuracy: {val_accuracy:.2%}",
        flush=True
    )

# --------------------------------------------------
# 7. Evaluate on test dataset
# --------------------------------------------------
test_accuracy = evaluate(test_loader)

print(f"\nFinal test accuracy: {test_accuracy:.2%}")

# --------------------------------------------------
# 8. Save results and trained model
# --------------------------------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

torch.save(
    model.cpu().state_dict(),
    results_dir / "cifar10_balanced_model.pth"
)

np.savetxt(
    results_dir / "cifar10_training_history.csv",
    np.array(history),
    delimiter=",",
    header="epoch,loss,train_accuracy,val_accuracy",
    comments=""
)

with open(
    results_dir / "cifar10_balanced_results.txt",
    "w"
) as file:
    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Test accuracy: {test_accuracy:.4f}\n")

print("\nModel and results saved successfully!")
