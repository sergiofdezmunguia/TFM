import sys
import pandas as pd
import numpy as np
import os
import json
import cv2
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
CLEANED_METADATA_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/metadata")
CLEANED_METADATA_CSV = os.path.join(CLEANED_METADATA_DIR, "cleaned_metadata.csv")
TRAIN_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "train_sample_ids.json")
VAL_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "val_sample_ids.json")
TEST_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "test_sample_ids.json")

# Directorio de los datos en bruto
RAW_DATA_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/gan_dataset-epsilon_greedy")
OBSTACLES_SUBDIR = "obstacle_maps"
PATHS_SUBDIR = "robot_paths"
GT_SUBDIR = "ground_truth"

# Directorio de salida para los datos preprocesados
PREPROCESSED_DATA_OUTPUT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/processed_for_model")

# Parámetros de preprocesamiento
IMG_HEIGHT = 256
IMG_WIDTH = 256
PATH_DRAW_THICKNESS = 1

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def preprocess_sample(obstacle_map_raw, robot_path_df_raw, gt_map_raw, target_h, target_w, path_thickness):
    """
    Preprocesa una muestra de datos, redimensionando las imágenes y dibujando la ruta del robot sobre el mapa de obstáculos.
    """

    # 1. Preprocesar el mapa de obstáculos (Canal 1)
    obstacle_map_processed = obstacle_map_raw.astype(np.float32)
    obstacle_map_processed = np.expand_dims(obstacle_map_processed, axis=-1)

    # 2. Preprocesar la ruta del robot (Canal 2)
    path_mask = np.zeros((target_h, target_w), dtype=np.float32)
    path_values = np.zeros((target_h, target_w), dtype=np.float32)

    if not robot_path_df_raw.empty and {'pos_i', 'pos_j', 'concentration'}.issubset(robot_path_df_raw.columns):
        points_data = []
        for _, row in robot_path_df_raw.iterrows():
            pos_i, pos_j, concentration = int(round(row['pos_i'])), int(round(row['pos_j'])), row['concentration']
            pt_i = np.clip(pos_i, 0, target_h - 1)
            pt_j = np.clip(pos_j, 0, target_w - 1)
            
            points_data.append(((pt_j, pt_i), concentration))

        for k in range(len(points_data) - 1):
            pt1, pt2 = points_data[k][0], points_data[k + 1][0]
            cv2.line(path_mask, pt1, pt2, 1, thickness=path_thickness)
            cv2.line(path_values, pt1, pt2, points_data[k][1], thickness=path_thickness)

    path_mask_processed = np.expand_dims(path_mask, axis=-1)
    path_values_processed = np.expand_dims(path_values, axis=-1)

    # 3. Apilar canales de entrada
    model_input_X = np.concatenate([obstacle_map_processed, path_mask_processed, path_values_processed], axis=-1)

    # 4. Preprocesar el mapa GT (Salida Y)
    gt_map_processed = gt_map_raw.astype(np.float32)
    model_output_Y = np.expand_dims(gt_map_processed, axis=-1)

    return model_input_X.astype(np.float32), model_output_Y.astype(np.float32)

def preprocess_and_save_split(df_split, split_name, raw_data_parent_dir, output_dir, roi_h, roi_w, path_thickness):
    """
    Preprocesa y guarda los datos de un split específico (train, val, test).
    """
    print(f"\nProcesando split: {split_name} ({len(df_split)} muestras)")
    
    output_split_dir_X = os.path.join(output_dir, split_name, "inputs")
    output_split_dir_Y = os.path.join(output_dir, split_name, "targets")
    os.makedirs(output_split_dir_X, exist_ok=True)
    os.makedirs(output_split_dir_Y, exist_ok=True)

    obstacles_dir = os.path.join(raw_data_parent_dir, OBSTACLES_SUBDIR)
    paths_dir = os.path.join(raw_data_parent_dir, PATHS_SUBDIR)
    gt_dir = os.path.join(raw_data_parent_dir, GT_SUBDIR)

    for index, row in tqdm(df_split.iterrows(), total=df_split.shape[0], desc=f"Preprocesando {split_name}"):
        sample_id = row['sample_id']
        path_num = row['path_number']
        
        obstacle_file = os.path.join(obstacles_dir, row['obstacle_map_file'])
        path_file = os.path.join(paths_dir, row['robot_path_file'])
        gt_file = os.path.join(gt_dir, row['ground_truth_file'])

        try:
            obstacle_map_raw = np.load(obstacle_file)
            robot_path_df_raw = pd.read_csv(path_file)
            gt_map_raw = np.load(gt_file)

            # Llamar a la función sin redimensionamiento
            input_X, output_Y = preprocess_sample(
                obstacle_map_raw, robot_path_df_raw, gt_map_raw,
                roi_h, roi_w, path_thickness
            )

            unique_file_id = f"{sample_id}_path_{path_num}"
            np.save(os.path.join(output_split_dir_X, f"{unique_file_id}_input.npy"), input_X)
            np.save(os.path.join(output_split_dir_Y, f"{unique_file_id}_target.npy"), output_Y)

        except Exception as e:
            print(f"ERROR procesando {sample_id}_path_{path_num} para {split_name}: {e}. Saltando esta muestra.")
            continue

def main():
    print("--- Iniciando Preprocesamiento de Datos para el Modelo (SIN REDIMENSIONAMIENTO) ---")
    
    try:
        cleaned_df = pd.read_csv(CLEANED_METADATA_CSV)
        train_sample_ids = set(load_json(TRAIN_IDS_JSON))
        val_sample_ids = set(load_json(VAL_IDS_JSON))
        test_sample_ids = set(load_json(TEST_IDS_JSON))
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los archivos de metadatos o IDs: {e}")
        sys.exit(1)

    train_df = cleaned_df[cleaned_df['sample_id'].isin(train_sample_ids)]
    val_df = cleaned_df[cleaned_df['sample_id'].isin(val_sample_ids)]
    test_df = cleaned_df[cleaned_df['sample_id'].isin(test_sample_ids)]

    print(f"Muestras (rutas) para Train: {len(train_df)}")
    print(f"Muestras (rutas) para Val:   {len(val_df)}")
    print(f"Muestras (rutas) para Test:  {len(test_df)}")

    # Usar ROI_HEIGHT y ROI_WIDTH en lugar de IMG_HEIGHT, IMG_WIDTH
    preprocess_and_save_split(train_df, "train", RAW_DATA_DIR, PREPROCESSED_DATA_OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, PATH_DRAW_THICKNESS)
    preprocess_and_save_split(val_df, "val", RAW_DATA_DIR, PREPROCESSED_DATA_OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, PATH_DRAW_THICKNESS)
    preprocess_and_save_split(test_df, "test", RAW_DATA_DIR, PREPROCESSED_DATA_OUTPUT_DIR, IMG_HEIGHT, IMG_WIDTH, PATH_DRAW_THICKNESS)

    print("\n--- Preprocesamiento de Datos Finalizado ---")
    print(f"Datos preprocesados guardados en: {PREPROCESSED_DATA_OUTPUT_DIR}")

if __name__ == "__main__":
    if not os.path.exists(CLEANED_METADATA_CSV): # ... (verificaciones como antes) ...
        print(f"ERROR FATAL: El archivo '{CLEANED_METADATA_CSV}' no existe.")
        sys.exit(1)
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"ERROR FATAL: El directorio RAW_DATA_PARENT_DIR '{RAW_DATA_DIR}' no existe.")
        sys.exit(1)
    main()