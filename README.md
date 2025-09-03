# Gas Dispersion Mapping using Conditional GANs

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sergiofdezmunguia/TFM)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Master's Thesis - Universidad Autónoma de Madrid**  
*Author: Sergio Fernández de Munguía*  
*Academic Year: 2024-2025*

## Overview

This repository contains the complete implementation for gas dispersion mapping using conditional Generative Adversarial Networks (cGANs). The system predicts complete gas concentration maps from partial robot sensor measurements in indoor environments.

### Key Features

- **Physics-based simulation**: Numerical gas diffusion with wind field modeling
- **Deep learning architecture**: U-Net Generator + PatchGAN Discriminator
- **Comprehensive evaluation**: IoU, PSNR, SSIM, and domain-specific metrics
- **Complete pipeline**: From data generation to model deployment
- **CUDA acceleration**: GPU-optimized simulation and training

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8+ GB RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sergiofdezmunguia/TFM.git
cd TFM
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
cd src/data_scripts
python full_data_gen.py
```

### Train Model

```bash
cd src/gan
python train.py --dataset ../../data/processed_for_model --epochs 200
```

## Repository Structure

```
├── src/                          # Source code
│   ├── data_scripts/            # Dataset generation
│   ├── gan/                     # GAN implementation
│   └── ros/                     # ROS integration
├── data/                         # Datasets and maps
│   ├── maps/                    # Environment maps
│   └── README.md                # Data documentation
├── utils/                        # Visualization tools
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Results

| Configuration | IoU Mean | PSNR | SSIM |
|---------------|----------|------|------|
| Without Wind  | 0.564    | 21.69| 0.836|
| With Wind     | 0.358    | 17.03| 0.741|

## Documentation

- **[Data Documentation](data/README.md)**: Dataset structure and generation
- **[Source Code Documentation](src/README.md)**: Code organization and APIs
- **[Thesis Figures](thesis_figures/README.md)**: Generated visualizations

## Usage Examples

### Basic Gas Simulation

```python
from src.data_scripts.diffusion_generator import generate_diffusion_map_roi

# Generate gas dispersion map
gas_map, obstacles = generate_diffusion_map_roi(
    map_subsection_np=map_data,
    source_coords_px_relative=(128, 128),
    timesteps=150000
)
```

### Model Prediction

```python
from src.gan.predict import load_model, predict_gas_map

# Load trained model
model = load_model('path/to/model.pth')

# Make prediction
prediction = predict_gas_map(model, robot_path, obstacle_map)
```

## Contributing

This is an academic research project. For questions or suggestions:

1. Open an issue on GitHub
2. Contact: sergio.fernandezm02@estudiante.uam.es

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Acknowledgments

- Universidad Autónoma de Madrid - EPS
- Master in Data Science
- Computer Engineer University Department

---

*This repository contains the complete implementation for the Master's Thesis "Gas Dispersion Mapping using Conditional GANs" at Universidad Autónoma de Madrid.*
