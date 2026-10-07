from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path(r"C:\Users\tseye\Datasets\IE4483\datasets")

IMAGE_SIZE = 128
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(
    DATA_ROOT / "train",
    transform=train_transform,
)

validation_dataset = datasets.ImageFolder(
    DATA_ROOT / "val",
    transform=validation_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

images, labels = next(iter(train_loader))

print("Class labels:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("One batch image shape:", images.shape)
print("One batch label shape:", labels.shape)