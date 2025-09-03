# Source Code - TFM Gas Dispersion Mapping

This directory contains the core implementation of the gas dispersion mapping system using conditional GANs.

## Directory Structure

### `data_scripts/` - Data Generation and Preprocessing
- **`diffusion_generator.py`** - Physics-based gas diffusion simulation using finite differences
- **`data_cleaning.py`** - Dataset filtering and quality validation  
- **`data_preprocess.py`** - Conversion of raw data to model input tensors

### `gan/` - Deep Learning Model Implementation  
- **`models.py`** - U-Net Generator and PatchGAN Discriminator architectures
- **`train.py`** - Complete training pipeline with adversarial loss
- **`evaluate.py`** - Model evaluation with domain-specific metrics
- **`hyperparameter_search.py`** - Automated hyperparameter optimization

### `ros/` - ROS Integration (Optional)
- Robot Operating System integration for real-world deployment
- Sensor data collection and processing nodes

## Key Features

### Physics-Based Simulation (`data_scripts/`)
- Numerical solution of diffusion equation with CUDA acceleration
- Localized wind field modeling with advection terms  
- Realistic indoor environments with complex obstacle layouts
- Batch processing for large-scale dataset generation

### Deep Learning Pipeline (`gan/`)
- **Generator**: U-Net architecture with skip connections
  - Input: 5-channel tensor (obstacles, path, detections, wind)
  - Output: 1-channel concentration map
- **Discriminator**: PatchGAN for local authenticity evaluation
- **Training**: Alternating optimization with combined adversarial + L1 loss
- **Evaluation**: IoU, PSNR, SSIM, and domain-specific metrics

## Usage Examples

### Generate Training Data
```bash
cd data_scripts
python diffusion_generator.py \
    --num_samples 1000 \
    --output_dir ../data/gan_dataset \
    --map_file ../data/maps/demo.pgm \
    --timesteps 200000
```

### Preprocess Data
```bash
python data_cleaning.py --input_dir ../data/gan_dataset
python data_preprocess.py --input_dir ../data/gan_dataset_cleaned
```

### Train Model
```bash
cd gan
python train.py \
    --train_dir ../data/processed_for_model/train \
    --val_dir ../data/processed_for_model/val \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3
```

### Hyperparameter Search
```bash
python hyperparameter_search.py \
    --dataset ../data/processed_for_model \
    --num_trials 30 \
    --output_dir ../models_outputs/hyperparam_search
```

## Performance Considerations

- **GPU Memory**: Training requires ~8GB GPU memory for batch_size=16
- **CUDA**: CuPy acceleration significantly speeds up data generation  
- **Dataset Size**: Full dataset generation can take 10+ hours on GPU
- **Training Time**: Complete training takes 6-12 hours depending on configuration

## Configuration

Key hyperparameters and their optimal ranges:

| Parameter | Range | Optimal |
|-----------|-------|---------|
| Learning Rate | 1e-4 to 1e-3 | 1e-3 |
| Lambda L1 | 25 to 100 | 50-75 |
| Generator Features | 32 or 64 | 64 (no wind), 32 (with wind) |
| Batch Size | 8 to 32 | 16 |

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- CuPy (for GPU acceleration)
- OpenCV, NumPy, Matplotlib
- See `../requirements.txt` for complete list
