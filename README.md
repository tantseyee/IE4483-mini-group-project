# IE4483 Mini Project (Option 2): Dogs vs. Cats Classification

IE4483 Artificial Intelligence and Data Mining, NTU. Completed by: Team 31 (Tan Tse Yee, Thoe Zi Lin, Jaslyn Lee Xin Pei)

This project trains image classifiers to tell whether an image contains a dog or a cat, using a dataset based on the Kaggle Dogs vs. Cats competition. It then extends the approach to multi-class classification on CIFAR-10, including an imbalanced-data setting.

## Team

| **Member** | **Name**           | **Responsibilities**                                                                                                                                |
| ---------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Member 1   | Tan Tse Yee        | Original data pipeline, preprocessing and augmentation, dataset check, Baseline v1 architecture and initial run                                     |
| Member 2   | Thoe Zi Lin        | Pipeline extensions, Baseline v1 retraining and Baseline v2, ResNet-18 model and experiments, test predictions and `submission.csv`, report figures |
| Member 3   | Jaslyn Lee Xin Pei | Dogs vs. Cats prediction error analysis, CIFAR-10 extension, class imbalance experiments                                                                                                     |

### Code contributions

| **File**                      | **Tse Yee**                                                                             | **Zi Lin**                                                                                                         | **Jaslyn** |
| ----------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------- |
| `src/check_data.py`           | Written                                                                                 | Path set-up comment                                                                                                |            |
| `src/data_pipeline.py`        | Transforms, augmentation, training and validation datasets and loaders                  | Test-set loader (`TestDataset`), `PRETRAINED` switch for ImageNet settings, `__main__` guard, path set-up comments |            |
| `src/baseline_cnn.py`         | Baseline v1 architecture, balanced subset selection, training loop, initial 3-epoch run | 20-epoch training, full seeding, history CSV output, `PRETRAINED` check, 20-epoch run                              |            |
| `src/baseline_cnn_v2.py`      |                                                                                         | Written and run                                                                                                    |            |
| `src/train_resnet.py`         |                                                                                         | Written; all six ResNet-18 runs                                                                                    |            |
| `src/predict.py`              |                                                                                         | Written; produced `submission.csv`                                                                                 |            |
| `src/plot_results.py`         |                                                                                         | Written                                                                                                            |            |
| `results/baseline_results.md` | Initial 3-epoch results                                                                 | Updated with the 20-epoch v1 and v2 results                                                                        |            |
| `results/resnet_results.md`   |                                                                                         | Written                                                                                                            |            |
| `README.md`                   |                                                                                         | Written                                                                                                            |            |
| `src/01_cifar10_data_prep.py` – `src/09_cifar10_extended_confusion_matrix.py` | | | Written and run; CIFAR-10 classification, imbalance experiments and evaluation |
| `src/10_dogs_cats_error_analysis.py` | | | Written and run; visual inspection of unlabelled test predictions |
| `src/11_dogs_cats_validation_analysis.py` | | | Written and run; labelled validation error analysis |
| `results/extended/` | | | CIFAR-10 10-epoch comparisons, confusion matrices, figures and reports |
| `results/dogs_cats_error_analysis/` and `results/dogs_cats_validation/` | | | Prediction examples, confusion matrix and validation results |

Individual contributions are described in full in the report.

## Project structure

```
main-file/
├── src/
│   ├── check_data.py        # Counts images in each split to verify the dataset
│   ├── data_pipeline.py     # Transforms, datasets and data loaders (train, val, test)
│   ├── baseline_cnn.py      # Baseline v1: 3-layer CNN trained from scratch
│   ├── baseline_cnn_v2.py   # Baseline v2: 4-layer CNN with BatchNorm and dropout
│   ├── train_resnet.py      # ResNet-18 (pretrained or random weights), two-stage training
│   ├── predict.py           # Predicts the 500 test images and writes submission.csv
│   ├── plot_results.py      # Makes the report figures from the history CSVs
│   ├── 01_cifar10_data_prep.py ... 09_cifar10_extended_confusion_matrix.py
│   ├── 10_dogs_cats_error_analysis.py       # Visual inspection of test predictions
│   └── 11_dogs_cats_validation_analysis.py  # Labelled validation error analysis
├── results/
│   ├── baseline_results.md              # Baseline results and observations
│   ├── resnet_results.md                # ResNet results and observations
│   ├── baseline_history_seed42.csv      # Per-epoch history, Baseline v1
│   ├── baseline_v2_history_seed42.csv   # Per-epoch history, Baseline v2
│   ├── resnet18_subset*_history_seed*.csv  # Per-epoch history, one file per ResNet run
│   ├── resnet18_subset*_best_seed*.pt      # Best model of each ResNet run (not in the repo)
│   ├── resnet18_subset_test_predictions.csv  # Test predictions with probabilities
│   ├── figures/                         # PNG figures for the report
│   ├── extended/                        # CIFAR-10 extended results and figures
│   ├── dogs_cats_error_analysis/        # Unlabelled test-set prediction examples
│   └── dogs_cats_validation/            # Labelled validation evaluation
├── notebooks/               # Exploratory notebooks
├── submission.csv           # Final test-set predictions (part d)
└── README.md

```

The dataset is **not** included in this repository. Each member keeps it locally (see Setup).

## Dataset

Download the dataset from the Google Drive link in the project brief. It should have this layout:

```
datasets/
├── train/
│   ├── cat/    # 10,000 images
│   └── dog/    # 10,000 images
├── val/
│   ├── cat/    # 2,500 images
│   └── dog/    # 2,500 images
└── test/       # 500 unlabelled images

```

Labels follow the assignment: **0 = cat, 1 = dog**. PyTorch's `ImageFolder` assigns these automatically because it sorts class folders alphabetically.

## Setup

### 1. Install requirements

Python 3.9 or later is recommended.

```
pip install torch torchvision numpy pillow matplotlib
# Additional packages needed for the Member 3 analysis scripts:
pip install tensorflow scikit-learn pandas
```

A GPU is strongly recommended for training (e.g. Google Colab or Kaggle Notebooks). The code also runs on a CPU, but much more slowly.

### 2. Set your dataset paths

**Before running anything**, open these two files and change `DATA_ROOT` to the `datasets` folder on your own computer:

- `src/data_pipeline.py`
- `src/check_data.py`

Examples:

```
DATA_ROOT = Path(r"C:\Users\yourname\Datasets\IE4483\datasets")   # Windows
DATA_ROOT = Path("/Users/yourname/Datasets/IE4483/datasets")      # Mac/Linux
DATA_ROOT = Path("/content/drive/MyDrive/IE4483/datasets")        # Google Colab
```

In `src/data_pipeline.py`, `TEST_ROOT` sets the location of the test images. By default it is `DATA_ROOT / "test"`. Only change it if your `test` folder is somewhere else.

Please don't commit your personal path changes when pushing shared code.

### 3. Choose the model settings

In `src/data_pipeline.py`, set `PRETRAINED` to match the model you are training:

| **`PRETRAINED`** | **Used for**                   | **Image size** | **Normalisation**                                              |
| ---------------- | ------------------------------ | -------------- | -------------------------------------------------------------- |
| `False`          | Baseline CNN (from scratch)    | 128 × 128      | mean 0.5, std 0.5 per channel                                  |
| `True`           | Pretrained model (e.g. ResNet) | 224 × 224      | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |

Pretrained models expect the same image size and normalisation as the ImageNet images they were originally trained on.

## How to run

Run all commands from the project root folder (`main-file`).

### Step 1: Check the dataset

```
python src/check_data.py
```

Expected output: 10,000 images in each training class, 2,500 in each validation class, and 500 test images.

### Step 2: Check the data pipeline

```
python src/data_pipeline.py
```

This prints the active settings, the class-to-label mapping, the number of training, validation and test images, the shape of one batch, and the first 10 test image IDs.

Expected batch shapes: `[32, 3, 128, 128]` for images with `PRETRAINED = False`, or `[32, 3, 224, 224]` with `PRETRAINED = True`, and `[32]` for labels.

Importing `data_pipeline` from another script does not run these checks.

### Step 3: Train the baseline CNNs

Set `PRETRAINED = False` in `src/data_pipeline.py`. Both baseline scripts stop with an error message if it is set to `True`.

**Baseline v1** (3 conv blocks):

```
python src/baseline_cnn.py
```

**Baseline v2** (4 conv blocks with BatchNorm and dropout):

```
python src/baseline_cnn_v2.py
```

Both train on the same balanced subset, with 2,000 training images and 500 validation images per class, and use identical settings apart from the architecture. Each prints the loss, training accuracy and validation accuracy for every epoch, and saves the per-epoch history to the `results` folder.

On a CPU each run takes roughly 30–60 minutes. On a Mac, prefix the command with `caffeinate -i` to stop the computer sleeping during training, e.g. `caffeinate -i python src/baseline_cnn.py`.

The main settings are at the top of each file:

| **Setting**            | **File**                                | **Default** |
| ---------------------- | --------------------------------------- | ----------- |
| `PRETRAINED`           | `data_pipeline.py`                      | `False`     |
| `BATCH_SIZE`           | `data_pipeline.py`                      | 32          |
| `TRAIN_PER_CLASS`      | `baseline_cnn.py`, `baseline_cnn_v2.py` | 2000        |
| `VALIDATION_PER_CLASS` | `baseline_cnn.py`, `baseline_cnn_v2.py` | 500         |
| `EPOCHS`               | `baseline_cnn.py`, `baseline_cnn_v2.py` | 20          |
| `SEED`                 | `baseline_cnn.py`, `baseline_cnn_v2.py` | 42          |

### Step 4: Train the ResNet-18 model

Set `PRETRAINED = True` in `src/data_pipeline.py` (224 × 224 images, ImageNet normalisation). `train_resnet.py` and `predict.py` stop with an error message if it is set to `False`.

```
python src/train_resnet.py
```

With ImageNet weights, training runs in two stages:

1. **Stage 1:** the pretrained layers are frozen and only the new final layer, Linear(512 → 2), is trained.
2. **Stage 2:** all layers are unfrozen and the whole network is fine-tuned with a much smaller learning rate.

It uses the same 4,000 training and 1,000 validation images as the baselines (seed 42), runs on the CPU, and downloads the ImageNet weights (about 45 MB) on the first run. Expect roughly 45–60 minutes per run on a laptop CPU.

The settings are in one block at the top of `src/train_resnet.py`:

| **Setting**                            | **Default**       | **Meaning**                                                                                     |
| -------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| `SEED`                                 | 42                | Random seed; also selects the training and validation subset                                    |
| `USE_SUBSET`                           | `True`            | `False` uses the full 20,000 / 5,000 images                                                     |
| `USE_IMAGENET_WEIGHTS`                 | `True`            | `False` trains ResNet-18 from random weights, for comparison                                    |
| `STAGE1_EPOCHS`, `STAGE1_LR`           | 3, 1e-3           | Stage 1 (frozen backbone)                                                                       |
| `STAGE2_EPOCHS`, `STAGE2_LR`           | 5, 1e-4           | Stage 2 (fine-tuning all layers)                                                                |
| `RANDOM_INIT_EPOCHS`, `RANDOM_INIT_LR` | 8, 1e-3           | Only used with random weights                                                                   |
| `RUN_NAME`                             | `resnet18_subset` | Prefix for the output files                                                                     |
| `OVERWRITE`                            | `False`           | If `False`, the script refuses to overwrite earlier results with the same `RUN_NAME` and `SEED` |

Each run saves its per-epoch history and its best model (by validation accuracy) to `results/`.

**Experiments reported** (change only the listed settings from the defaults):

| **Run**             | **Settings changed**           | **`RUN_NAME`**                |
| ------------------- | ------------------------------ | ----------------------------- |
| Main model          | none                           | `resnet18_subset`             |
| High fine-tuning LR | `STAGE2_LR = 1e-3`             | `resnet18_subset_ft_lr1e-3`   |
| High head LR        | `STAGE1_LR = 1e-2`             | `resnet18_subset_head_lr1e-2` |
| Random weights      | `USE_IMAGENET_WEIGHTS = False` | `resnet18_subset_random_init` |
| Seed 2              | `SEED = 123`                   | `resnet18_subset`             |
| Seed 3              | `SEED = 2026`                  | `resnet18_subset`             |

### Step 5: Predict the test set and write `submission.csv`

Keep `PRETRAINED = True`, then run:

```
python src/predict.py
```

This loads `results/resnet18_subset_best_seed42.pt` (the main model), predicts all 500 test images, and checks the result: 500 rows, no duplicate IDs, labels only 0 or 1, and, if `sampleSubmission.csv` is in the project folder, IDs that match it. It writes:

- `submission.csv` in the project folder, with the `id,label` columns required by the assignment
- `results/resnet18_subset_test_predictions.csv`, which adds each image's probability and confidence (used for the error analysis)

It also prints the 10 least confident predictions. To use a different trained model, change `RUN_NAME` and `SEED` at the top of `src/predict.py`.

### Step 6: Make the figures

```
python src/plot_results.py
```

This reads the history CSVs in `results/` and saves six PNG figures to `results/figures/`. Any figure whose CSV files are missing is skipped.

| **Figure**                  | **Shows**                                                                    |
| --------------------------- | ---------------------------------------------------------------------------- |
| `fig1_model_comparison.png` | Validation accuracy per epoch for both baselines and both ResNet-18 versions |
| `fig2_best_accuracy.png`    | Best validation accuracy of each model                                       |
| `fig3_resnet_training.png`  | Main ResNet-18 model: training and validation accuracy and loss              |
| `fig4_learning_rates.png`   | ResNet-18 learning-rate comparison                                           |
| `fig5_seeds.png`            | Main ResNet-18 model over three seeds                                        |
| `fig6_baselines.png`        | Baselines v1 and v2: training vs validation accuracy                         |

### Step 7: Dogs vs. Cats error analysis (part e; Jaslyn Lee Xin Pei)

Two complementary analyses were performed:

- **Unlabelled test set:** `src/10_dogs_cats_error_analysis.py` uses `results/resnet18_subset_test_predictions.csv` and the 500 images in `datasets/test/` to display high- and low-confidence predictions. Because the test images have no ground-truth labels, these are **visual assessments, not confirmed test-set errors**.
- **Labelled validation set:** `src/11_dogs_cats_validation_analysis.py` loads a fine-tuned ResNet-18 checkpoint and evaluates the 5,000 labelled images in `datasets/val/`, producing a classification report, confusion matrix, predictions CSV and example images.

For the validation script, place `resnet18_subset_best_seed123.pt` at `checkpoints/resnet18_subset_best_seed123.pt` relative to the repository root. The checkpoint is **not committed to GitHub**. The script uses 224 × 224 images and ImageNet normalisation; it loads the saved weights without retraining. Run both commands from the repository root:

```bash
python src/10_dogs_cats_error_analysis.py
python src/11_dogs_cats_validation_analysis.py
```

The seed-123 checkpoint recorded **98.70%** on its 1,000-image validation subset. When evaluated on all 5,000 labelled validation images, it achieved **98.62% accuracy** (4,931 correct, 69 incorrect). These figures refer to different validation-set sizes and should not be treated as identical evaluations. The test prediction CSV was produced by the group's main model, whereas the labelled validation analysis used the seed-123 checkpoint.

Saved outputs are in `results/dogs_cats_error_analysis/` and `results/dogs_cats_validation/`.

### Step 8: CIFAR-10 and class imbalance (parts g and h; Jaslyn Lee Xin Pei)

The CIFAR-10 experiments use 10 image classes and a separate CNN with a 10-class output layer. The balanced split contains 40,000 training images, 10,000 validation images and 10,000 test images. For the imbalanced experiments, the **cat (class 3)** and **dog (class 5)** training classes are reduced from 4,000 to 400 examples each; validation and test distributions remain unchanged. The two mitigation approaches are **class weighting** and **random oversampling**.

Run commands **from the repository root**, in this order:

```bash
# Prepare the balanced/imbalanced CIFAR-10 data and mitigation inputs
python src/01_cifar10_data_prep.py
python src/02_cifar10_imbalance_prep.py
python src/03_cifar10_class_weighting.py
python src/04_cifar10_oversampling.py

# Initial 3-epoch CNN experiments and confusion matrices
python src/05_cifar10_pytorch.py
python src/06_cifar10_imbalance_experiments.py
python src/07_cifar10_confusion_matrix.py

# Extended 10-epoch experiments and confusion matrices
python src/08_cifar10_extended_experiments.py
python src/09_cifar10_extended_confusion_matrix.py
```

The preparation scripts create local intermediate files such as `cifar10_imbalanced.npz`, `cifar10_oversampled.npz` and `cifar10_class_weights.npy`. Later scripts expect these files in the **current working directory**. Generated datasets and trained model files (`.npz`, `.npy`, `.pt`, `.pth`) are not included in this repository; rerunning the corresponding stages is necessary to recreate them. TensorFlow/Keras is used to obtain CIFAR-10, while PyTorch is used for CNN training.

**Extended results (10 epochs):**

| Training condition | Test accuracy | Macro-F1 | Cat recall | Dog recall |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 52.80% | 0.5140 | 16.00% | 41.80% |
| Imbalanced | 48.99% | 0.4390 | 0.10% | 2.70% |
| Class weighting | 52.21% | 0.5129 | 13.30% | 66.20% |
| Oversampling | **56.85%** | **0.5657** | **31.20%** | 58.60% |

Oversampling achieved the highest overall accuracy and macro-F1 in these runs. Class weighting produced the highest dog recall. These are results from a limited experiment, not proof that one mitigation method is universally best. The 10-epoch results, per-class reports, confusion matrices and accuracy comparison plot are saved in `results/extended/`. The earlier 3-epoch outputs can be regenerated using scripts 05–07.

**Reproducibility note:** The CIFAR-10 experiments use their own training settings and should not be conflated with the Dogs vs. Cats baseline/ResNet experiments described above. Reported results are from the completed runs; results may vary across hardware and random seeds.

## Pre-processing and augmentation

|                        | **Training**                                   | **Validation / Test** |
| ---------------------- | ---------------------------------------------- | --------------------- |
| Resize                 | 128 × 128 (baseline) or 224 × 224 (pretrained) | Same as training      |
| Random horizontal flip | Yes                                            | No                    |
| Random rotation (±10°) | Yes                                            | No                    |
| Normalisation          | Depends on `PRETRAINED` (see Setup, step 3)    | Same as training      |

Augmentation is applied to the training set only. Test images use the validation transform, and are converted to RGB so greyscale or transparent images do not cause errors.

## Reproducibility

- A fixed random seed (`SEED = 42`) is used for subset selection, model initialisation and data shuffling.
- Python, NumPy and PyTorch (CPU and GPU) are all seeded, and cuDNN is set to deterministic mode.
- All reported runs used the CPU, where results are fully reproducible. Two ResNet runs that share seed 42 and the same stage 1 settings produced identical stage 1 results to every decimal place.
- The main ResNet-18 model was trained with three seeds (42, 123, 2026) and is reported as mean ± standard deviation. A different seed also selects a different training and validation subset.

## Results

All runs used the CPU, 4,000 training images and a validation subset of 1,000 images. Unless a seed is given, the seed is 42, so every model in the first table was trained and validated on exactly the same images.

| **Model**                                                         | **Epochs** | **Best validation accuracy** |
| ----------------------------------------------------------------- | ---------- | ---------------------------- |
| Baseline v1, initial run (undertrained)                           | 3          | 59.90%                       |
| Baseline v1                                                       | 20         | 69.00%                       |
| Baseline v2                                                       | 20         | 75.80%                       |
| ResNet-18, random weights                                         | 8          | 74.70%                       |
| ResNet-18, ImageNet weights, after stage 1 only (frozen backbone) | 3          | 97.80%                       |
| **ResNet-18, ImageNet weights, fine-tuned (main model)**          | 3 + 5      | **98.50%**                   |

ResNet-18 learning-rate and seed experiments:

| **Run**                                       | **Best validation accuracy**                      |
| --------------------------------------------- | ------------------------------------------------- |
| Main model (stage 1 LR 1e-3, stage 2 LR 1e-4) | 98.50%                                            |
| Stage 2 LR 1e-3                               | 97.80% (final epoch 96.80%)                       |
| Stage 1 LR 1e-2                               | 98.40%                                            |
| Main model, seeds 42 / 123 / 2026             | 98.50% / 98.70% / 99.30%, **mean 98.83% ± 0.42%** |

Full tables and discussion are in `results/baseline_results.md` and `results/resnet_results.md`.

## Submission

`submission.csv` was produced by `src/predict.py` from the main ResNet-18 model (`results/resnet18_subset_best_seed42.pt`, 98.50% validation accuracy). It contains one row per test image with two columns:

| **Column** | **Description**                                                                 |
| ---------- | ------------------------------------------------------------------------------- |
| `id`       | Image ID, taken from the test image filename and matching the sample submission |
| `label`    | Predicted class: 1 = dog, 0 = cat                                               |

## References

1. Kaggle Dogs vs. Cats: [https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)
2. CIFAR-10: [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)