import numpy as np
import pandas as pd
import yaml
import os
import shutil
from tqdm import tqdm
import cv2


from src.simulation_utils import (generate_robot_path, 
                                  get_free_pgm_threshold,
                                  DEFAULT_FREE_THRESH_PROB, 
                                  DEFAULT_OCCUPIED_THRESH_PROB)

# CONFIGURACIÓN PARA LA AUGMENTATION
PROJECT_ROOT = os.path.expanduser("~/uni/master/tfm/TFM")
ORIGINAL_DATASET_NAME = "gan_dataset-epsilon_greedy"
ORIGINAL_PARENT_DIR = os.path.join(PROJECT_ROOT, "data", ORIGINAL_DATASET_NAME)
ORIGINAL_METADATA_FILE = os.path.join(ORIGINAL_PARENT_DIR, "metadata_reconstructed.csv")
ORIGINAL_GT_DIR = os.path.join(ORIGINAL_PARENT_DIR, "ground_truth")
ORIGINAL_OBSTACLES_DIR = os.path.join(ORIGINAL_PARENT_DIR, "obstacle_maps")
ORIGINAL_MAPS_DIR = os.path.join(PROJECT_ROOT, "data", "maps") 
ORIGINAL_MAP_NAME = "demo"

AUGMENTED_DATASET_NAME = "gan_dataset_augmented"
AUG_PARENT_DIR = os.path.join(PROJECT_ROOT, "data", AUGMENTED_DATASET_NAME)
AUG_GT_DIR = os.path.join(AUG_PARENT_DIR, "ground_truth")
AUG_OBSTACLES_DIR = os.path.join(AUG_PARENT_DIR, "obstacle_maps")
AUG_PATHS_DIR = os.path.join(AUG_PARENT_DIR, "robot_paths")
AUG_METADATA_FILE = os.path.join(AUG_PARENT_DIR, "metadata.csv")

# Parámetros para las nuevas rutas
NUM_NEW_PATHS_PER_SAMPLE_ID = 10
PATH_GEN_PARAMS = {
    "algorithm": "epsilon_greedy",
    "epsilon": 0.3,
    "max_steps": 60,
    "noise_std_dev": 0.015,
    "min_distance_from_source": 55
}

# Crear directorios de salida
os.makedirs(AUG_GT_DIR, exist_ok=True)
os.makedirs(AUG_OBSTACLES_DIR, exist_ok=True)
os.makedirs(AUG_PATHS_DIR, exist_ok=True)

def augment_existing_dataset():
    print(f"Augmentando rutas para el dataset: {ORIGINAL_PARENT_DIR}")
    print(f"Salida en: {AUG_PARENT_DIR}")

    try:
        original_meta_df = pd.read_csv(ORIGINAL_METADATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Metadata original no encontrado: {ORIGINAL_METADATA_FILE}"); return
    
    map_pgm_file = os.path.join(ORIGINAL_MAPS_DIR, ORIGINAL_MAP_NAME + ".pgm")
    map_yaml_file = os.path.join(ORIGINAL_MAPS_DIR, ORIGINAL_MAP_NAME + ".yaml")
    try:
        with open(map_yaml_file, 'r') as f: map_yaml = yaml.safe_load(f)
        map_resolution = map_yaml['resolution']
        free_thresh_pgm_val = get_free_pgm_threshold(map_yaml.get('free_thresh', DEFAULT_FREE_THRESH_PROB))
        map_pgm_full = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
        if map_pgm_full is None: raise ValueError("PGM no cargado")
    except Exception as e:
        print(f"ERROR: Cargando mapa PGM base '{map_pgm_file}': {e}"); return

    augmented_metadata_entries = []
    unique_samples = original_meta_df.drop_duplicates(subset=['sample_id'])

    for _, sample_data in tqdm(unique_samples.iterrows(), total=len(unique_samples), desc="Procesando Samples"):
        sample_id = sample_data['sample_id']
        gt_file = sample_data['ground_truth_file']
        obs_file = sample_data['obstacle_map_file']

        roi_orig_i = sample_data.get('roi_origin_px_i', np.nan)
        roi_orig_j = sample_data.get('roi_origin_px_j', np.nan)
        src_rel_i = sample_data.get('source_relative_px_i', np.nan)
        src_rel_j = sample_data.get('source_relative_px_j', np.nan)

        if np.isnan(src_rel_i) or np.isnan(src_rel_j):
            try:
                gt_map_temp = np.load(os.path.join(ORIGINAL_GT_DIR, gt_file))
                src_rel_i, src_rel_j = np.unravel_index(np.argmax(gt_map_temp), gt_map_temp.shape)
            except Exception:
                print(f"  Skipping {sample_id}: no se pudo obtener/inferir la fuente relativa."); continue
        
        source_coords_in_roi = (src_rel_i, src_rel_j)

        # Copiar GT y Obstáculos
        try:
            shutil.copy2(os.path.join(ORIGINAL_GT_DIR, gt_file), os.path.join(AUG_GT_DIR, gt_file))
            shutil.copy2(os.path.join(ORIGINAL_OBSTACLES_DIR, obs_file), os.path.join(AUG_OBSTACLES_DIR, obs_file))
        except Exception as e:
            print(f"  Skipping {sample_id}: error copiando archivos base: {e}"); continue

        # Cargar los mapas para generar rutas
        try:
            current_gt = np.load(os.path.join(AUG_GT_DIR, gt_file))
            current_obs_mask = np.load(os.path.join(AUG_OBSTACLES_DIR, obs_file)).astype(bool)
        except Exception as e:
            print(f"  Skipping {sample_id}: error cargando archivos copiados: {e}"); continue
        
        roi_h, roi_w = current_obs_mask.shape
        roi_pgm_values = None
        if not (np.isnan(roi_orig_i) or np.isnan(roi_orig_j)):
            r_i, r_j = int(roi_orig_i), int(roi_orig_j)
            if r_i + roi_h <= map_pgm_full.shape[0] and r_j + roi_w <= map_pgm_full.shape[1]:
                roi_pgm_values = map_pgm_full[r_i : r_i + roi_h, r_j : r_j + roi_w]
            else:
                print(f"  WARN {sample_id}: Coordenadas ROI fuera de mapa PGM. Usando fallback para PGM values.")
                roi_pgm_values = np.full_like(current_obs_mask, 255, dtype=np.uint8)
                roi_pgm_values[current_obs_mask] = 0
        else:
            print(f"  WARN {sample_id}: Sin info de origen ROI. Usando fallback para PGM values.")
            roi_pgm_values = np.full_like(current_obs_mask, 255, dtype=np.uint8)
            roi_pgm_values[current_obs_mask] = 0

        for i in range(NUM_NEW_PATHS_PER_SAMPLE_ID):
            path_df = generate_robot_path(
                obstacle_map=current_obs_mask,
                concentration_map=current_gt,
                resolution=map_resolution,
                source_coords_px=source_coords_in_roi,
                free_pgm_min_value=free_thresh_pgm_val,
                roi_map_pgm_values=roi_pgm_values,
                **PATH_GEN_PARAMS
            )

            if path_df is not None and not path_df.empty:
                new_path_filename = f"{sample_id}_path_aug_{i}.csv"
                path_df.to_csv(os.path.join(AUG_PATHS_DIR, new_path_filename), index=False, float_format='%.6f')
                augmented_metadata_entries.append({
                    'sample_id': sample_id,
                    'path_number': f"aug_{i}",
                    'map_name': sample_data.get('map_name', ORIGINAL_MAP_NAME),
                    'roi_origin_px_i': roi_orig_i, 'roi_origin_px_j': roi_orig_j,
                    'source_relative_px_i': src_rel_i, 'source_relative_px_j': src_rel_j,
                    'ground_truth_file': gt_file,
                    'obstacle_map_file': obs_file,
                    'robot_path_file': new_path_filename,
                    'num_path_steps': len(path_df)
                })
    
    if augmented_metadata_entries:
        aug_df = pd.DataFrame(augmented_metadata_entries)
        aug_df.to_csv(AUG_METADATA_FILE, index=False)
        print(f"\nMetadata aumentado ({len(aug_df)} rutas) guardado en: {AUG_METADATA_FILE}")
    else:
        print("\nNo se generaron rutas aumentadas.")
    print("--- Augmentación Finalizada ---")

if __name__ == "__main__":
    augment_existing_dataset()