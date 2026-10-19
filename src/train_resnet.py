import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import models

from data_pipeline import train_dataset, validation_dataset, BATCH_SIZE, PRETRAINED

if not PRETRAINED:
    raise ValueError(
        "ResNet needs PRETRAINED = True (224 x 224 images, ImageNet normalisation). "
        "Change it at the top of src/data_pipeline.py and run again."
    )


SEED = 2026

# True  -> same 2,000 + 2,000 training and 500 + 500 validation subset as the baselines (fair comparison, faster).
# False -> full dataset: 20,000 training and 5,000 validation images.
USE_SUBSET = True
TRAIN_PER_CLASS = 2000
VALIDATION_PER_CLASS = 500

# True  -> start from ImageNet weights and train in two stages (main model).
# False -> start from random weights (no pretraining), for comparison only.
USE_IMAGENET_WEIGHTS = True

# Used when USE_IMAGENET_WEIGHTS = True.
# Stage 1: frozen backbone, train the final layer only.
STAGE1_EPOCHS = 3
STAGE1_LR = 1e-3
# Stage 2: unfreeze everything and fine-tune.
STAGE2_EPOCHS = 5
STAGE2_LR = 1e-4

# Used when USE_IMAGENET_WEIGHTS = False.
RANDOM_INIT_EPOCHS = 8
RANDOM_INIT_LR = 1e-3

# Name used for the output files. Change it for each experiment so earlier results are not overwritten.
RUN_NAME = "resnet18_subset"

OVERWRITE = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")
print("Using device:", device)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
HISTORY_FILE = RESULTS_DIR / f"{RUN_NAME}_history_seed{SEED}.csv"
BEST_MODEL_FILE = RESULTS_DIR / f"{RUN_NAME}_best_seed{SEED}.pt"

if not OVERWRITE and (BEST_MODEL_FILE.exists() or HISTORY_FILE.exists()):
    raise FileExistsError(
        f"Results for RUN_NAME = '{RUN_NAME}' and SEED = {SEED} already exist in {RESULTS_DIR}.\n"
        "Change RUN_NAME or SEED for a new experiment, "
        "or set OVERWRITE = True to replace them."
    )


def make_balanced_subset(dataset, images_per_class):
    """Select the same number of cat and dog images (identical to the baselines)."""
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


if USE_SUBSET:
    train_data = make_balanced_subset(train_dataset, TRAIN_PER_CLASS)
    validation_data = make_balanced_subset(validation_dataset, VALIDATION_PER_CLASS)
else:
    train_data = train_dataset
    validation_data = validation_dataset

print(f"Training images: {len(train_data)} | Validation images: {len(validation_data)}")

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


def build_model(use_imagenet_weights):
    """ResNet-18 (ImageNet or random weights); the final layer is replaced for 2 classes."""
    if use_imagenet_weights:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_backbone_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) every layer except the final layer."""
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, loss_function, backbone_frozen):
    model.train()

    if backbone_frozen:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += labels.size(0)

    return total_loss / total_images, correct_predictions / total_images


def evaluate(model, loader, loss_function):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

    return total_loss / total_images, correct_predictions / total_images


model = build_model(USE_IMAGENET_WEIGHTS).to(device)
print("Starting weights:", "ImageNet (pretrained)" if USE_IMAGENET_WEIGHTS else "random (no pretraining)")
loss_function = nn.CrossEntropyLoss()

history = []
best = {"validation_accuracy": 0.0, "epoch": 0, "stage": 0}
epoch_counter = 0


def run_stage(stage, epochs, learning_rate, backbone_frozen, description):
    global epoch_counter

    if epochs <= 0:
        return

    set_backbone_trainable(model, trainable=not backbone_frozen)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(trainable_parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(
        f"\n=== Stage {stage}: {description} "
        f"| trainable parameters: {count_trainable(model):,} | starting LR: {learning_rate} ==="
    )

    for _ in range(epochs):
        epoch_counter += 1
        start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, loss_function, backbone_frozen
        )
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_function)
        scheduler.step()

        seconds = time.time() - start_time

        if validation_accuracy > best["validation_accuracy"]:
            best.update(
                validation_accuracy=validation_accuracy,
                epoch=epoch_counter,
                stage=stage,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch_counter,
                    "stage": stage,
                    "run_name": RUN_NAME,
                    "seed": SEED,
                    "imagenet_weights": USE_IMAGENET_WEIGHTS,
                },
                BEST_MODEL_FILE,
            )

        history.append([
            epoch_counter, stage, current_lr,
            train_loss, train_accuracy, validation_loss, validation_accuracy,
            round(seconds, 1),
        ])

        print(
            f"Epoch {epoch_counter} (stage {stage}) | LR: {current_lr:.2e} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.2%} | "
            f"Val loss: {validation_loss:.4f} | Val acc: {validation_accuracy:.2%} | "
            f"{seconds:.0f}s"
        )


if USE_IMAGENET_WEIGHTS:
    run_stage(
        stage=1, epochs=STAGE1_EPOCHS, learning_rate=STAGE1_LR, backbone_frozen=True,
        description="frozen backbone, training final layer only",
    )
    run_stage(
        stage=2, epochs=STAGE2_EPOCHS, learning_rate=STAGE2_LR, backbone_frozen=False,
        description="fine-tuning all layers",
    )
else:
    run_stage(
        stage=1, epochs=RANDOM_INIT_EPOCHS, learning_rate=RANDOM_INIT_LR, backbone_frozen=False,
        description="training all layers from random weights",
    )

with open(HISTORY_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "stage", "learning_rate",
        "train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
        "seconds",
    ])
    writer.writerows(history)

last_epochs = [row[6] for row in history[-5:]]
print(
    f"\nBest validation accuracy: {best['validation_accuracy']:.2%} "
    f"(epoch {best['epoch']}, stage {best['stage']})"
)
print(f"Average of last {len(last_epochs)} epochs: {sum(last_epochs) / len(last_epochs):.2%}")
print(f"History saved to: {HISTORY_FILE}")
print(f"Best model saved to: {BEST_MODEL_FILE}")