import re
from pathlib import Path
 
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Change this before running:
# Set DATA_ROOT to the "datasets" folder on your computer.
# This folder should contain the "train", "val" and "test" subfolders.
#   Windows example: Path(r"C:\Users\yourname\Datasets\IE4483\datasets")
#   Mac/Linux example: Path("/Users/yourname/Datasets/IE4483/datasets")
#   Colab example: Path("/content/drive/MyDrive/IE4483/datasets")
# Keep the r before the quotes on Windows so the backslashes work.

DATA_ROOT = Path(" ")
TEST_ROOT = DATA_ROOT / "test"

# Choose the model type before running:
#   PRETRAINED = False -> baseline CNN trained from scratch (Tse Yee)
#   PRETRAINED = True  -> pretrained model such as ResNet (Zi Lin)
PRETRAINED = False
 
if PRETRAINED:
    IMAGE_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)  # ImageNet values
    STD = (0.229, 0.224, 0.225)
else:
    IMAGE_SIZE = 128
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
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

def get_image_id(file):
    """Take the number from a filename, e.g. "123.jpg" -> 123."""
    numbers = re.findall(r"\d+", file.stem)
    if not numbers:
        raise ValueError(f"No number found in test filename: {file.name}")
    return int(numbers[-1])


class TestDataset(Dataset):
    """Unlabelled test images. Returns (image, image_id) instead of (image, label)."""

    def __init__(self, folder, transform=None):
        image_extensions = {".jpg", ".jpeg", ".png"}
        self.files = [
            file for file in Path(folder).iterdir()
            if file.suffix.lower() in image_extensions
        ]
        self.files.sort(key=get_image_id)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = Image.open(file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, get_image_id(file)

test_dataset = TestDataset(TEST_ROOT, transform=validation_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("Pretrained settings:", PRETRAINED, "| Image size:", IMAGE_SIZE)
    print("Class labels:", train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Test images:", len(test_dataset))
    print("One batch image shape:", images.shape)
    print("One batch label shape:", labels.shape)
    print("First 10 test image IDs:", [get_image_id(f) for f in test_dataset.files[:10]])