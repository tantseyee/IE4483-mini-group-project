# ResNet-18 Results

## Common setup

These settings are the same for every run below unless the table says otherwise.

- Model: ResNet-18, final layer replaced with Linear(512 → 2)
- Image size: 224 × 224, ImageNet normalisation (`PRETRAINED = True`)
- Augmentation (training only): random horizontal flip, random rotation (±10°)
- Training subset: 2,000 cats and 2,000 dogs (4,000 images)
- Validation subset: 500 cats and 500 dogs (1,000 images)
- Batch size: 32
- Loss: cross-entropy
- Optimiser: Adam, with a cosine learning-rate schedule within each stage
- Training (ImageNet weights): stage 1 = 3 epochs with the backbone frozen (final layer only, 1,026 trainable parameters); stage 2 = 5 epochs fine-tuning all layers (11,177,538 trainable parameters)
- Device: CPU (MacBook)

With seed 42, the training and validation subsets are exactly the same images as for both baseline CNNs, so all seed-42 results can be compared directly. The only input difference is that ResNet-18 sees the images at 224 × 224 with ImageNet normalisation, while the baselines use 128 × 128 with mean and std 0.5.

## Summary

| Run | Starting weights | Stage 1 LR | Stage 2 LR | Seed | Best after stage 1 | Best validation accuracy | Final val. loss | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Main model** | ImageNet | 1e-3 | 1e-4 | 42 | 97.80% | **98.50%** (epoch 8) | 0.0613 | 45 min |
| High fine-tuning LR | ImageNet | 1e-3 | **1e-3** | 42 | 97.80% | 97.80% (epoch 2); final 96.80% | 0.0854 | 47 min |
| High head LR | ImageNet | **1e-2** | 1e-4 | 42 | 98.30% | 98.40% (epoch 8) | 0.0697 | 49 min |
| Random weights | **Random** | 1e-3 (single stage, 8 epochs, all layers) | | 42 | n/a | 74.70% (epoch 8) | 0.5134 | 62 min |
| Main model, seed 2 | ImageNet | 1e-3 | 1e-4 | **123** | 98.10% | 98.70% (epoch 8) | 0.0364 | 54 min |
| Main model, seed 3 | ImageNet | 1e-3 | 1e-4 | **2026** | 99.10% | 99.30% (epoch 8) | 0.0232 | 57 min |

**Main model over 3 seeds (42, 123, 2026): 98.83% ± 0.42%** (mean ± sample standard deviation). After stage 1 alone: 98.33% ± 0.68%.

Changing the seed also changes which 4,000 training and 1,000 validation images are selected, so the spread across seeds reflects both the random initialisation of the new layer and the choice of data subset.

## Comparison with the baselines (seed 42, same images)

| Model | Best validation accuracy |
|---|---:|
| Baseline v1 (3 conv blocks, from scratch, 20 epochs) | 69.00% |
| Baseline v2 (4 conv blocks + BatchNorm + dropout, from scratch, 20 epochs) | 75.80% |
| ResNet-18, random weights (8 epochs) | 74.70% |
| ResNet-18, ImageNet weights, frozen backbone only (stage 1, 3 epochs) | 97.80% |
| **ResNet-18, ImageNet weights, fine-tuned (stages 1 + 2, 8 epochs)** | **98.50%** |

## Per-epoch results

### Main model (seed 42)

| Epoch | Stage | LR | Train loss | Train acc. | Val. loss | Val. acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00e-03 | 0.1743 | 92.63% | 0.0813 | 97.00% |
| 2 | 1 | 7.50e-04 | 0.0666 | 98.03% | 0.0736 | 97.80% |
| 3 | 1 | 2.50e-04 | 0.0580 | 97.98% | 0.0640 | 97.70% |
| 4 | 2 | 1.00e-04 | 0.0983 | 96.43% | 0.1639 | 93.30% |
| 5 | 2 | 9.05e-05 | 0.0434 | 98.28% | 0.0743 | 97.70% |
| 6 | 2 | 6.55e-05 | 0.0248 | 99.08% | 0.0784 | 97.40% |
| 7 | 2 | 3.45e-05 | 0.0130 | 99.58% | 0.0769 | 97.80% |
| 8 | 2 | 9.55e-06 | 0.0071 | 99.80% | 0.0613 | **98.50%** |

### High fine-tuning LR (stage 2 LR = 1e-3)

| Epoch | Stage | LR | Train loss | Train acc. | Val. loss | Val. acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00e-03 | 0.1743 | 92.63% | 0.0813 | 97.00% |
| 2 | 1 | 7.50e-04 | 0.0666 | 98.03% | 0.0736 | **97.80%** |
| 3 | 1 | 2.50e-04 | 0.0580 | 97.98% | 0.0640 | 97.70% |
| 4 | 2 | 1.00e-03 | 0.2626 | 89.43% | 0.1328 | 94.90% |
| 5 | 2 | 9.05e-04 | 0.1469 | 94.18% | 0.2247 | 90.10% |
| 6 | 2 | 6.55e-04 | 0.1133 | 95.33% | 0.1463 | 93.30% |
| 7 | 2 | 3.45e-04 | 0.0687 | 97.35% | 0.1019 | 96.50% |
| 8 | 2 | 9.55e-05 | 0.0356 | 98.68% | 0.0854 | 96.80% |

### High head LR (stage 1 LR = 1e-2)

| Epoch | Stage | LR | Train loss | Train acc. | Val. loss | Val. acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.00e-02 | 0.1178 | 95.33% | 0.0888 | 97.10% |
| 2 | 1 | 7.50e-03 | 0.0492 | 98.28% | 0.0668 | 98.20% |
| 3 | 1 | 2.50e-03 | 0.0402 | 98.35% | 0.0660 | 98.30% |
| 4 | 2 | 1.00e-04 | 0.1332 | 95.85% | 0.1814 | 93.50% |
| 5 | 2 | 9.05e-05 | 0.0527 | 97.83% | 0.1228 | 97.10% |
| 6 | 2 | 6.55e-05 | 0.0296 | 99.03% | 0.0946 | 97.70% |
| 7 | 2 | 3.45e-05 | 0.0177 | 99.40% | 0.0940 | 97.70% |
| 8 | 2 | 9.55e-06 | 0.0093 | 99.75% | 0.0697 | **98.40%** |

### Random weights (no pretraining)

| Epoch | LR | Train loss | Train acc. | Val. loss | Val. acc. |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00e-03 | 0.7156 | 58.28% | 0.7135 | 57.00% |
| 2 | 9.62e-04 | 0.6639 | 61.93% | 0.7193 | 55.00% |
| 3 | 8.54e-04 | 0.6513 | 63.53% | 0.6693 | 61.90% |
| 4 | 6.91e-04 | 0.6257 | 65.40% | 0.6477 | 65.30% |
| 5 | 5.00e-04 | 0.6061 | 67.28% | 0.6939 | 65.30% |
| 6 | 3.09e-04 | 0.5789 | 70.28% | 0.5701 | 70.90% |
| 7 | 1.46e-04 | 0.5373 | 73.50% | 0.5623 | 71.00% |
| 8 | 3.81e-05 | 0.5041 | 75.48% | 0.5134 | **74.70%** |

### Main model, seeds 123 and 2026 (validation accuracy)

| Epoch | Stage | Seed 42 | Seed 123 | Seed 2026 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 97.00% | 97.30% | 98.40% |
| 2 | 1 | 97.80% | 97.50% | 98.30% |
| 3 | 1 | 97.70% | 98.10% | 99.10% |
| 4 | 2 | 93.30% | 97.70% | 97.30% |
| 5 | 2 | 97.70% | 98.50% | 99.10% |
| 6 | 2 | 97.40% | 98.50% | 99.10% |
| 7 | 2 | 97.80% | 98.50% | 99.20% |
| 8 | 2 | **98.50%** | **98.70%** | **99.30%** |

Full histories are in `results/resnet18_subset*_history_seed*.csv`.

## Observations

**Pretraining is the main reason for the high accuracy.** With the same architecture, data and number of epochs (8), ResNet-18 reached 98.50% from ImageNet weights but only 74.70% from random weights, a gap of 23.8 percentage points. (The random-weights run trained all layers in a single stage at a learning rate of 1e-3, since freezing random features would be meaningless.) After 8 epochs, ResNet-18 from random weights was at about the level Baseline v2 reached after 20 epochs (75.80%). Its validation accuracy was still rising at the end (71.00% to 74.70% in the last epoch), so it would probably improve with longer training, but nowhere near the pretrained model within a comparable training time. The benefit comes mainly from features already learned on over a million ImageNet images, not from the larger network.

**The pretrained features are almost enough on their own.** Training only the new final layer (1,026 parameters) for 3 epochs gave 97.80%. Fine-tuning all 11 million parameters added a further 0.7 points (98.50%). This gain was consistent across all three seeds (+0.7, +0.6 and +0.2 points), but it is small and comes at roughly twice the training time per epoch.

**A small fine-tuning learning rate is necessary.** With a stage 2 learning rate of 1e-3 instead of 1e-4, validation accuracy fell from 97.80% to 90.10% within two epochs of unfreezing, and only recovered to 96.80% by the end, below the frozen-backbone result. Large updates damage the useful pretrained features faster than the model can relearn them from 4,000 images. With 1e-4, the drop after unfreezing was brief and the model ended above its stage 1 result.

**The head learning rate matters much less.** Using 1e-2 for stage 1 learned slightly faster (98.30% vs 97.80% after stage 1), but ended at 98.40% vs 98.50%. That 0.1-point difference is a single image out of 1,000, smaller than the 0.42-point spread between seeds, so it is not a meaningful difference. The higher rate gave no accuracy benefit and a slightly higher final validation loss (0.0697 vs 0.0613), so the original rate of 1e-3 was kept.

**A temporary drop after unfreezing is expected.** In every run that started from ImageNet weights, validation accuracy dipped at epoch 4, the first epoch after unfreezing (to 93.30%, 97.70% and 97.30% across the three seeds, and to 94.90% and 93.50% in the two learning-rate runs). When the backbone is unfrozen, all layers start changing at once and the batch-norm layers begin updating their statistics again, which briefly disrupts the pretrained features before the model recovers.

**Mild overfitting by the end.** In the main model, training accuracy reached 99.80% against 98.50% validation accuracy, and the training loss (0.0071) fell well below the validation loss (0.0613). More epochs would be unlikely to help.

**Reproducibility.** The first three epochs of the main run and the high fine-tuning LR run are identical to every decimal place, because both use seed 42 and differ only from stage 2 onwards. This confirms that the CPU runs are fully reproducible with the fixed seed. Across three seeds, the main model achieved 98.83% ± 0.42%.