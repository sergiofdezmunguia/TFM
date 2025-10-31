#!/usr/bin/env python3
"""
Script to create a validation dataset with representative ground truth samples
for thesis supervisor review and model validation.

Selects a subset of ground truth data for validation purposes.
"""

import os
import shutil
import numpy as np
import random
from pathlib import Path

# Configuration
SAMPLES_PER_DATASET = 15  # Manageable sample size for validation
RANDOM_SEED = 42

def select_validation_samples(dataset_dir, output_dir, num_samples=15):
    """Select representative ground truth samples from a dataset"""
    
    gt_dir = os.path.join(dataset_dir, "ground_truth")
    
    if not os.path.exists(gt_dir):
        print(f"Warning: {gt_dir} not found")
        return []
    
    # Get all ground truth files
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
    
    if len(gt_files) == 0:
        print(f"No ground truth files found in {gt_dir}")
        return []
    
    # Select representative samples (every N-th sample for even distribution)
    total_files = len(gt_files)
    step = max(1, total_files // num_samples)
    
    selected_files = []
    for i in range(0, min(total_files, num_samples * step), step):
        if i < len(gt_files):
            selected_files.append(gt_files[i])
    
    # If we need more samples, add some random ones
    if len(selected_files) < num_samples:
        remaining = [f for f in gt_files if f not in selected_files]
        additional_needed = num_samples - len(selected_files)
        selected_files.extend(random.sample(remaining, min(additional_needed, len(remaining))))
    
    # Copy selected ground truth files
    copied_files = []
    for gt_file in selected_files[:num_samples]:
        src_gt = os.path.join(gt_dir, gt_file)
        dst_gt = os.path.join(output_dir, gt_file)
        
        try:
            shutil.copy2(src_gt, dst_gt)
            copied_files.append(gt_file)
        except Exception as e:
            print(f"Error copying {gt_file}: {e}")
    
    return copied_files

def main():
    random.seed(RANDOM_SEED)
    
    base_dir = "data"
    output_dir = "data/validation_samples"
    
    # Create clean output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    datasets = [
        "gan_dataset_wind",  # With wind - primary validation set
        "gan_dataset-epsilon_greedy"  # Without wind - baseline comparison
    ]
    
    all_copied = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset} not found, skipping...")
            continue
            
        print(f"Selecting validation samples from {dataset}...")
        
        # Create subdirectory for this dataset
        dataset_output = os.path.join(output_dir, dataset.split('_')[-1])  # wind/epsilon_greedy
        os.makedirs(dataset_output, exist_ok=True)
        
        copied = select_validation_samples(
            dataset_path, 
            dataset_output, 
            SAMPLES_PER_DATASET
        )
        
        all_copied[dataset] = copied
        print(f"Selected {len(copied)} samples from {dataset}")
    
    # Print summary
    total_files = sum(len(files) for files in all_copied.values())
    print(f"\nValidation dataset created:")
    print(f"Total ground truth samples: {total_files}")
    print(f"Location: {output_dir}")
    
    for dataset, files in all_copied.items():
        print(f"  {dataset}: {len(files)} samples")
    
    # Calculate total size
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', output_dir], capture_output=True, text=True)
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"Total size: {size}")
    except:
        pass

if __name__ == "__main__":
    main()