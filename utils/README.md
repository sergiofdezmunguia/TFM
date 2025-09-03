# Utility Scripts

This directory contains utility scripts for data visualization and figure generation used in the thesis.

## Scripts

### Visualization Tools
- `create_gas_comparison_simple.py` - Generate gas dispersion comparison figures
- `create_architecture_diagram.py` - Create GAN architecture diagrams  
- `create_discarded_examples_individual.py` - Visualize data filtering examples
- `enhance_prediction_plots.py` - Enhanced model prediction visualizations

### Data Analysis
- `create_comparison_tables.py` - Generate LaTeX tables from results
- `consolidate_csv.py` - Merge and process experimental results

### Helper Scripts
- `run_path_comparison.sh` - Automated figure generation pipeline

## Usage

Each script can be run independently:

```bash
python create_gas_comparison_simple.py
python create_architecture_diagram.py
```

Most scripts automatically save outputs to the appropriate directories (`../thesis_figures/`, `../tfm_memoria/img/`, etc.).
