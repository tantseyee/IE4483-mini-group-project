# Baseline CNN Results

## Common setup

These settings are the same for every run below, so differences in accuracy come only from the number of epochs or the architecture.

- Image size: 128 × 128
- Normalisation: mean 0.5, std 0.5 per channel (`PRETRAINED = False`)
- Augmentation (training only): random horizontal flip, random rotation (±10°)
- Training subset: 2,000 cats and 2,000 dogs (4,000 images)
- Validation subset: 500 cats and 500 dogs (1,000 images)
- Batch size: 32
- Loss: cross-entropy
- Optimiser: Adam, learning rate 0.001 (constant)
- Random seed: 42 (Python, NumPy and PyTorch seeded; cuDNN deterministic)
- Device: CPU. The initial 3-epoch run was on Tse Yee's computer; the 20-epoch runs of v1 and v2 were on Zi Lin's MacBook.

## Models

| Model | Architecture | Script |
|---|---|---|
| Baseline v1 | 3 conv blocks (Conv + ReLU + MaxPool), 16 → 32 → 64 feature maps, adaptive average pooling, Linear(64 → 2) | `src/baseline_cnn.py` |
| Baseline v2 | 4 conv blocks (Conv + BatchNorm + ReLU + MaxPool), 16 → 32 → 64 → 128 feature maps, adaptive average pooling, Dropout(0.5), Linear(128 → 2) | `src/baseline_cnn_v2.py` |

## Summary

| Run | Epochs | Best validation accuracy | Average of last 5 epochs | Final training loss |
|---|---:|---:|---:|---:|
| Baseline v1, initial run | 3 | 59.90% (epoch 2) | n/a | 0.6599 |
| Baseline v1 | 20 | 69.00% (epoch 18) | 67.32% | 0.5668 |
| Baseline v2 | 20 | **75.80% (epoch 19)** | **73.40%** | **0.4898** |

The average of the last 5 epochs (epochs 16–20) is reported alongside the best epoch, because validation accuracy fluctuates by several points between epochs. A single best epoch partly reflects this noise.

## Baseline v1: initial run (3 epochs)

| Epoch | Loss | Training Accuracy | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.6862 | 54.33% | 55.40% |
| 2 | 0.6671 | 59.67% | 59.90% |
| 3 | 0.6599 | 60.32% | 57.80% |

## Baseline v1: 20 epochs

Full history: `results/baseline_history_seed42.csv`

The first three epochs differ very slightly from the initial run. The initial run was made on a different computer, and small numerical differences between machines and library versions are expected even with the same seed. Results are fully reproducible on the same computer.

| Epoch | Loss | Training Accuracy | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.6860 | 54.40% | 55.40% |
| 2 | 0.6670 | 59.55% | 59.40% |
| 3 | 0.6599 | 60.63% | 58.10% |
| 4 | 0.6567 | 60.88% | 56.10% |
| 5 | 0.6562 | 61.18% | 57.70% |
| 6 | 0.6505 | 61.58% | 58.10% |
| 7 | 0.6442 | 62.55% | 59.10% |
| 8 | 0.6389 | 63.40% | 63.50% |
| 9 | 0.6297 | 63.65% | 62.30% |
| 10 | 0.6258 | 64.53% | 60.00% |
| 11 | 0.6194 | 65.70% | 65.10% |
| 12 | 0.6128 | 65.72% | 63.50% |
| 13 | 0.6086 | 66.90% | 65.60% |
| 14 | 0.6055 | 66.80% | 66.90% |
| 15 | 0.5975 | 67.62% | 66.60% |
| 16 | 0.5880 | 68.23% | 66.10% |
| 17 | 0.5823 | 69.42% | 68.40% |
| 18 | 0.5807 | 68.73% | **69.00%** |
| 19 | 0.5714 | 70.33% | 64.20% |
| 20 | 0.5668 | 70.73% | 68.90% |

## Baseline v2: 20 epochs

Full history: `results/baseline_v2_history_seed42.csv`

| Epoch | Loss | Training Accuracy | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.6927 | 59.13% | 58.90% |
| 2 | 0.6475 | 62.42% | 57.20% |
| 3 | 0.6346 | 64.48% | 65.60% |
| 4 | 0.6145 | 66.35% | 66.50% |
| 5 | 0.6160 | 66.67% | 60.90% |
| 6 | 0.5957 | 68.83% | 60.20% |
| 7 | 0.5813 | 69.10% | 68.70% |
| 8 | 0.5655 | 71.20% | 67.20% |
| 9 | 0.5562 | 71.47% | 71.60% |
| 10 | 0.5498 | 72.20% | 70.50% |
| 11 | 0.5398 | 72.92% | 71.00% |
| 12 | 0.5508 | 73.30% | 73.00% |
| 13 | 0.5289 | 74.28% | 64.70% |
| 14 | 0.5273 | 74.15% | 66.50% |
| 15 | 0.5148 | 74.50% | 69.70% |
| 16 | 0.5149 | 75.65% | 74.10% |
| 17 | 0.5050 | 75.33% | 72.60% |
| 18 | 0.5120 | 74.70% | 73.80% |
| 19 | 0.4975 | 76.12% | **75.80%** |
| 20 | 0.4898 | 76.12% | 70.70% |

## Observations

**Number of epochs.** The initial 3-epoch run reached only 59.90% validation accuracy, close to random guessing (50%), with the loss barely falling below 0.693. This showed the model was undertrained rather than limited by its architecture. Training the same v1 model for 20 epochs raised the best validation accuracy to 69.00% (epoch 18).

**Underfitting in v1.** The training loss was still falling at epoch 20, and training and validation accuracy stayed close (70.73% vs 68.90%). This suggests v1 is underfitting: its limited capacity, rather than overfitting, is the main constraint.

**Effect of the architecture changes (v2).** To test whether the baseline was limited by its architecture, v2 adds batch normalisation after each convolutional layer, a fourth convolutional block (64 → 128 feature maps) and dropout (p = 0.5) before the final layer. All other settings were kept identical to v1. v2 reached a best validation accuracy of 75.80% (epoch 19) compared with 69.00% for v1, and averaged 73.40% over the last five epochs compared with 67.32%. Its training loss also fell further (0.4898 vs 0.5668). v2 also learned faster, passing 65% validation accuracy by epoch 3.

**Fluctuation.** Validation accuracy fluctuates between epochs in both models (for example, v2 dropped from 73.00% to 64.70% at epoch 13). The constant learning rate of 0.001 likely contributes to this.

**Conclusion.** Neither model shows overfitting, as training and validation accuracy remain close and the loss is still falling at epoch 20. Small CNNs trained from scratch on 4,000 images appear to be limited mainly by model capacity and the amount of training data. This motivates the use of a pretrained backbone (e.g. ResNet), which has already learned general visual features from over a million ImageNet images.