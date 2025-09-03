# Data Directory

This directory contains the datasets used for training and evaluation.

## Structure

- `maps/` - Environment maps in PGM format
- `gan_dataset_wind/` - Dataset with wind effects (gitignored)
- `gan_dataset_augmented/` - Augmented dataset (gitignored)  
- `processed_for_model/` - Preprocessed tensors (gitignored)
- `metadata/` - Dataset metadata files (gitignored)

## Note

Large data files are excluded from git due to size constraints. 
Use the data generation scripts in `src/data_scripts/` to recreate datasets.
