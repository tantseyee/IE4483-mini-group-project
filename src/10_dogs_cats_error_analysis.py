
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "resnet18_subset_test_predictions.csv"
TEST_DIR = BASE_DIR / "datasets" / "test"
OUTPUT_DIR = BASE_DIR / "results" / "dogs_cats_error_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
df = pd.read_csv(CSV_PATH)

print("Total predictions:", len(df))
print("\nPredicted class counts:")
print(df["predicted_class"].value_counts())

print("\nConfidence summary:")
print(df["confidence"].describe())

# Select uncertain and confident predictions
uncertain = df.nsmallest(5, "confidence")
confident = df.nlargest(5, "confidence")

# Save selected predictions
selected = pd.concat([
    uncertain.assign(group="Low confidence"),
    confident.assign(group="High confidence")
])

selected.to_csv(
    OUTPUT_DIR / "selected_test_predictions.csv",
    index=False
)

# Display selected images
fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for row, (group_name, samples) in enumerate([
    ("Low confidence", uncertain),
    ("High confidence", confident)
]):
    for col, (_, sample) in enumerate(samples.iterrows()):
        image_id = int(sample["id"])
        image_path = TEST_DIR / f"{image_id}.jpg"

        ax = axes[row, col]

        if image_path.exists():
            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))
        else:
            ax.text(0.5, 0.5, "Image not found",
                    ha="center", va="center")

        ax.set_title(
            f"ID: {image_id}\n"
            f"Predicted: {sample['predicted_class']}\n"
            f"Confidence: {sample['confidence']:.2%}",
            fontsize=10
        )
        ax.axis("off")

    axes[row, 0].set_ylabel(
        group_name, fontsize=12
    )

plt.suptitle(
    "Dogs vs Cats: High- and Low-Confidence Test Predictions",
    fontsize=14
)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_prediction_examples.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nResults saved to:", OUTPUT_DIR)
