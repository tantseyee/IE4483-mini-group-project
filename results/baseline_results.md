# Baseline CNN Results

## Setup
- Model: Simple CNN with 3 convolutional layers
- Image size: 128 × 128
- Training subset: 2,000 cats and 2,000 dogs
- Validation subset: 500 cats and 500 dogs
- Batch size: 32
- Optimiser: Adam
- Learning rate: 0.001
- Epochs: 3
- Device: CPU

## Results

| Epoch | Loss | Training Accuracy | Validation Accuracy |
|---|---:|---:|---:|
| 1 | 0.6862 | 54.33% | 55.40% |
| 2 | 0.6671 | 59.67% | 59.90% |
| 3 | 0.6599 | 60.32% | 57.80% |

## Initial observation
The baseline CNN learned better than random guessing, reaching its best validation accuracy of 59.90% in Epoch 2. 