import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import pandas as pd

class GasDiffusionDataset(Dataset):
    def __init__(self, metadata_df, input_dir_X, target_dir_Y, original_paths_csv_dir,
                 apply_augmentation=True, h_flip_prob=0.5, v_flip_prob=0.5):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.input_dir_X = input_dir_X
        self.target_dir_Y = target_dir_Y
        self.original_paths_csv_dir = original_paths_csv_dir
        self.apply_augmentation = apply_augmentation
        self.h_flip_prob = h_flip_prob if apply_augmentation else 0.0
        self.v_flip_prob = v_flip_prob if apply_augmentation else 0.0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()

        try:
            row = self.metadata_df.iloc[idx]
            sample_id = str(row['sample_id']).strip()
            path_id = str(row['path_number']).strip()
            robot_path_csv_filename = str(row['robot_path_file']).strip()
            
            base_filename = f"{sample_id}_path_{path_id}"
            input_X_path = os.path.join(self.input_dir_X, f"{base_filename}_input.npy")
            output_Y_path = os.path.join(self.target_dir_Y, f"{base_filename}_target.npy")

            input_X_np = np.load(input_X_path).astype(np.float32)
            output_Y_np = np.load(output_Y_path).astype(np.float32)

            # --- Carga de coordenadas de la ruta (lógica necesaria) ---
            robot_path_coords_px = []
            path_csv_full_path = os.path.join(self.original_paths_csv_dir, robot_path_csv_filename)
            if os.path.exists(path_csv_full_path):
                path_df = pd.read_csv(path_csv_full_path)
                if 'pos_j' in path_df.columns and 'pos_i' in path_df.columns:
                    coords = zip(path_df['pos_j'].values, path_df['pos_i'].values)
                    # Asegurarse de que las coordenadas son válidas antes de convertir a int
                    robot_path_coords_px = [(int(x), int(y)) for x, y in coords if not (np.isnan(x) or np.isnan(y))]
            
            # --- Lógica de aumentación que también afecta a las coordenadas ---
            img_h, img_w = input_X_np.shape[0], input_X_np.shape[1]
            if self.apply_augmentation:
                if random.random() < self.h_flip_prob:
                    input_X_np = np.ascontiguousarray(np.fliplr(input_X_np))
                    output_Y_np = np.ascontiguousarray(np.fliplr(output_Y_np))
                    if robot_path_coords_px: robot_path_coords_px = [(img_w - 1 - x, y) for x, y in robot_path_coords_px]
                if random.random() < self.v_flip_prob:
                    input_X_np = np.ascontiguousarray(np.flipud(input_X_np))
                    output_Y_np = np.ascontiguousarray(np.flipud(output_Y_np))
                    if robot_path_coords_px: robot_path_coords_px = [(x, img_h - 1 - y) for x, y in robot_path_coords_px]
            
            input_X_tensor = torch.from_numpy(input_X_np.transpose((2, 0, 1)))
            output_Y_tensor = torch.from_numpy(output_Y_np.transpose((2, 0, 1)))
            
            # --- Devolver los 5 elementos que train.py espera ---
            return input_X_tensor, output_Y_tensor, robot_path_coords_px, sample_id, path_id
        
        except Exception as e:
            # Si algo falla, devuelve None para que collate_fn lo salte
            print(f"ADVERTENCIA: Saltando muestra en el índice {idx} por error: {e}")
            return None