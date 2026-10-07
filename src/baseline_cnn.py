import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from data_pipeline import train_dataset, validation_dataset, BATCH_SIZE


# Use the same random selection each time, so results are easier to compare.
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Your computer is using CPU, so start with a smaller balanced dataset.
TRAIN_PER_CLASS = 2000
VALIDATION_PER_CLASS = 500
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def make_balanced_subset(dataset, images_per_class):
    """Select the same number of cat and dog images."""
    cat_indices = [
        index for index, label in enumerate(dataset.targets)
        if label == 0
    ]
    dog_indices = [
        index for index, label in enumerate(dataset.targets)
        if label == 1
    ]

    selected_cats = random.sample(cat_indices, images_per_class)
    selected_dogs = random.sample(dog_indices, images_per_class)

    return Subset(dataset, selected_cats + selected_dogs)


small_train_dataset = make_balanced_subset(train_dataset, TRAIN_PER_CLASS)
small_validation_dataset = make_balanced_subset(
    validation_dataset,
    VALIDATION_PER_CLASS,
)

train_loader = DataLoader(
    small_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    small_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


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
            nn.Linear(64, 2),
        )

    def forward(self, images):
        images = self.features(images)
        return self.classifier(images)


model = BaselineCNN().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def evaluate():
    model.eval()

    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_images += labels.size(0)

    return correct_predictions / total_images


for epoch in range(EPOCHS):
    model.train()

    correct_predictions = 0
    total_images = 0
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
        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    training_accuracy = correct_predictions / total_images
    validation_accuracy = evaluate()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train accuracy: {training_accuracy:.2%} | "
        f"Validation accuracy: {validation_accuracy:.2%}"
    )