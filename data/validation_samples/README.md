# Validation Samples Dataset

This directory contains a curated set of ground truth samples used for model validation and thesis supervisor review.

## Overview

- **Total samples**: 30 ground truth files (~7.7MB)
- **Purpose**: Model validation and academic review
- **Selection**: Representative samples from main training datasets

## Dataset Structure

```
validation_samples/
├── wind/                    # 15 samples from wind-enabled dataset
│   ├── sample_00XXX_gt.npy
│   └── ...
└── greedy/                  # 15 samples from epsilon-greedy baseline
    ├── sample_00XXX_gt.npy
    └── ...
```

## Sample Selection Criteria

- **Even distribution**: Samples selected at regular intervals across the full dataset
- **Representative coverage**: Both wind and no-wind scenarios included
- **Size optimized**: Small enough for Git repository (~7.7MB total)
- **Validation focus**: Sufficient for model performance verification

## File Format

- **Format**: NumPy arrays (.npy files)
- **Dimensions**: 256x256 pixels
- **Data type**: Float32
- **Value range**: 0.0 to 1.0 (normalized gas concentrations)

## Usage

### Load a sample in Python:
```python
import numpy as np

# Load ground truth sample
gt_data = np.load('validation_samples/wind/sample_00100_gt.npy')

# Basic statistics
print(f"Shape: {gt_data.shape}")
print(f"Max concentration: {np.max(gt_data):.4f}")
print(f"Mean concentration: {np.mean(gt_data):.4f}")
```

### Visualize samples:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Wind scenario
plt.subplot(1, 2, 1)
wind_gt = np.load('validation_samples/wind/sample_00100_gt.npy')
plt.imshow(wind_gt, cmap='viridis')
plt.title('With Wind')
plt.colorbar()

# No wind scenario  
plt.subplot(1, 2, 2)
greedy_gt = np.load('validation_samples/greedy/sample_00100_gt.npy')
plt.imshow(greedy_gt, cmap='viridis')
plt.title('No Wind (Epsilon-Greedy)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Source Datasets

- **wind/**: Selected from `gan_dataset_wind` (with wind field modeling)
- **greedy/**: Selected from `gan_dataset-epsilon_greedy` (baseline, no wind)

## Generation Script

To regenerate this validation dataset:

```bash
cd utils/
python create_validation_dataset.py
```

## Academic Use

This validation dataset is provided for:
- **Thesis supervisor review**
- **Model performance verification**  
- **Academic evaluation**
- **Research reproducibility**

The samples represent the ground truth data used to validate the conditional GAN models described in the thesis.

---

*Generated automatically from training datasets for academic review and validation purposes.*