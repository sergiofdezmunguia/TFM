# Thesis Figures

This directory contains generated figures for the thesis document. All figures are created programmatically to ensure reproducibility and consistency.

## Generated Figures

### Gas Dispersion Analysis
- `path_comparison_epsilon_greedy.png` - Robot exploration strategy comparison
- `path_comparison_random_walk.png` - Random walk baseline comparison

### Model Architecture
- `gan_complete_architecture.png` - Complete GAN architecture diagram

### Data Processing Examples
- `discarded_example_*.png` - Examples of filtered training samples

## Generation Scripts

Located in `../utils/`:
- `create_gas_comparison_simple.py` - Gas dispersion visualizations
- `create_architecture_diagram.py` - Model architecture diagrams
- `create_discarded_examples_individual.py` - Data filtering examples
- `enhance_prediction_plots.py` - Result analysis plots

## Usage

To regenerate all figures:

```bash
cd ../utils
python create_gas_comparison_simple.py
python create_architecture_diagram.py
python create_discarded_examples_individual.py
```

All figures use the `viridis` colormap for consistency and are optimized for LaTeX document integration.

## Note

Generated images are excluded from git to reduce repository size. Run the respective scripts to recreate figures locally.
