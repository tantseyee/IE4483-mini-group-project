from pathlib import Path

DATA_ROOT = Path(r"C:\Users\tseye\Datasets\IE4483\datasets")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(folder: Path) -> int:
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

for split in ["train", "val"]:
    for label in ["cat", "dog"]:
        folder = DATA_ROOT / split / label
        print(f"{split}/{label}: {count_images(folder)} images")

print(f"test: {count_images(DATA_ROOT / 'test')} images")