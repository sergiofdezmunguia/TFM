import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class GasDiffusionDataset(Dataset):
    def __init__(self, metadata_df, input_dir_X, target_dir_Y, 
                 apply_augmentation=True, aug_h_flip_prob=0.4, aug_v_flip_prob=0.4):
        """
        Dataset personalizado para cargar mapas de difusión de gas y aplicar augmentation.

        Args:
            metadata_df (pd.DataFrame): DataFrame filtrado (ej. train_df) con la metadata.
            input_dir_X (str): Directorio de los archivos _input.npy (HxWx3).
            target_dir_Y (str): Directorio de los archivos _target.npy (HxWx1).
            apply_augmentation (bool): Flag general para activar/desactivar augmentation.
            aug_h_flip_prob (float): Probabilidad de aplicar flip horizontal (0.0 a 1.0).
            aug_v_flip_prob (float): Probabilidad de aplicar flip vertical (0.0 a 1.0).
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.input_dir_X = input_dir_X
        self.target_dir_Y = target_dir_Y
        
        self.apply_augmentation = apply_augmentation
        self.h_flip_prob = aug_h_flip_prob if apply_augmentation else 0.0
        self.v_flip_prob = aug_v_flip_prob if apply_augmentation else 0.0

        if not (0.0 <= self.h_flip_prob <= 1.0 and 0.0 <= self.v_flip_prob <= 1.0):
            raise ValueError("Las probabilidades de flip deben estar entre 0.0 y 1.0")

    def __len__(self):
        """Devuelve el número total de muestras en el dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Carga y devuelve una muestra del dataset en el índice `idx`.
        Aplica augmentation si está configurado.
        """
        if torch.is_tensor(idx): 
            idx = idx.tolist()

        row = self.metadata_df.iloc[idx]
        sample_id = row['sample_id']
        path_num = int(row['path_number'])
        
        unique_file_id = f"{sample_id}_path_{path_num}"
        
        input_X_path = os.path.join(self.input_dir_X, f"{unique_file_id}_input.npy")
        output_Y_path = os.path.join(self.target_dir_Y, f"{unique_file_id}_target.npy")

        try:
            input_X_np = np.load(input_X_path).astype(np.float32)
            output_Y_np = np.load(output_Y_path).astype(np.float32)
        except Exception as e:
            print(f"ERROR CRÍTICO cargando datos para {unique_file_id} (índice {idx}): {e}")

            if idx == 0:
                raise RuntimeError(f"Fallo al cargar la muestra inicial 0 ({unique_file_id}): {e}") from e
            print(f"ADVERTENCIA: Fallo al cargar muestra {idx}. Intentando cargar la muestra 0 en su lugar.")
            return self.__getitem__(0)

        # Data Augmentation
        if self.apply_augmentation:
            # Flip Horizontal Aleatorio
            if random.random() < self.h_flip_prob:
                input_X_np = np.ascontiguousarray(np.fliplr(input_X_np))
                output_Y_np = np.ascontiguousarray(np.fliplr(output_Y_np))

            # Flip Vertical Aleatorio
            if random.random() < self.v_flip_prob:
                input_X_np = np.ascontiguousarray(np.flipud(input_X_np))
                output_Y_np = np.ascontiguousarray(np.flipud(output_Y_np))

        # Convertir NumPy arrays HWC a Tensores PyTorch CHW 
        input_X_tensor = torch.from_numpy(input_X_np.transpose((2, 0, 1)))
        output_Y_tensor = torch.from_numpy(output_Y_np.transpose((2, 0, 1)))

        return input_X_tensor, output_Y_tensor