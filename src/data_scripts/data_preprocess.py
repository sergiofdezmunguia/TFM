import sys
import pandas as pd
import numpy as np
import os
import json
import cv2
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
CLEANED_METADATA_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/metadata/wind_cleaned")
CLEANED_METADATA_CSV = os.path.join(CLEANED_METADATA_DIR, "cleaned_metadata.csv")
TRAIN_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "train_sample_ids.json")
VAL_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "val_sample_ids.json")
TEST_IDS_JSON = os.path.join(CLEANED_METADATA_DIR, "test_sample_ids.json")

# Directorio de los datos en bruto
RAW_DATA_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/gan_dataset_wind")
OBSTACLES_SUBDIR = "obstacle_maps"
PATHS_SUBDIR = "robot_paths"
GT_SUBDIR = "ground_truth"
WIND_VY_SUBDIR = "wind_fields_vy"
WIND_VX_SUBDIR = "wind_fields_vx"

# Directorio de salida para los datos preprocesados
PREPROCESSED_DATA_OUTPUT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/processed_for_model_wind")

# Parámetros de preprocesamiento
IMG_HEIGHT = 256
IMG_WIDTH = 256
PATH_DRAW_THICKNESS = 3

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def preprocess_sample(obstacle_map_raw, robot_path_df_raw, wind_vy_map_raw, wind_vx_map_raw):
    # Canal 0: Mapa de obstáculos
    obstacle_map_processed = obstacle_map_raw.astype(np.float32)

    # Crear lienzos para los 4 canales de la ruta
    path_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    path_gas_values = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    path_wind_vy_values = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    path_wind_vx_values = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    if not robot_path_df_raw.empty and {'pos_i', 'pos_j', 'concentration'}.issubset(robot_path_df_raw.columns):
        points_data = []
        for _, row in robot_path_df_raw.iterrows():
            pos_i, pos_j = int(round(row['pos_i'])), int(round(row['pos_j']))
            pt_i = np.clip(pos_i, 0, IMG_HEIGHT - 1)
            pt_j = np.clip(pos_j, 0, IMG_WIDTH - 1)
            
            concentration = row['concentration']
            wind_vy_reading = wind_vy_map_raw[pt_i, pt_j]
            wind_vx_reading = wind_vx_map_raw[pt_i, pt_j]
            
            points_data.append(((pt_j, pt_i), concentration, wind_vy_reading, wind_vx_reading))

        for k in range(len(points_data) - 1):
            pt1, pt2 = points_data[k][0], points_data[k + 1][0]
            gas_val, wind_vy_val, wind_vx_val = points_data[k][1], points_data[k][2], points_data[k][3]

            cv2.line(path_mask, pt1, pt2, 1.0, thickness=PATH_DRAW_THICKNESS)
            cv2.line(path_gas_values, pt1, pt2, float(gas_val), thickness=PATH_DRAW_THICKNESS)
            cv2.line(path_wind_vy_values, pt1, pt2, float(wind_vy_val), thickness=PATH_DRAW_THICKNESS)
            cv2.line(path_wind_vx_values, pt1, pt2, float(wind_vx_val), thickness=PATH_DRAW_THICKNESS)

    # Apilar los 5 canales de entrada
    model_input_X = np.stack([
        obstacle_map_processed,
        path_mask,
        path_gas_values,
        path_wind_vy_values,
        path_wind_vx_values
    ], axis=-1)

    return model_input_X

def preprocess_and_save_split(df_split, split_name):
    print(f"\nProcesando split: {split_name} ({len(df_split)} muestras)")
    
    output_dir_X = os.path.join(PREPROCESSED_DATA_OUTPUT_DIR, split_name, "inputs")
    output_dir_Y = os.path.join(PREPROCESSED_DATA_OUTPUT_DIR, split_name, "targets")
    os.makedirs(output_dir_X, exist_ok=True); os.makedirs(output_dir_Y, exist_ok=True)

    obstacles_dir = os.path.join(RAW_DATA_DIR, OBSTACLES_SUBDIR)
    paths_dir = os.path.join(RAW_DATA_DIR, PATHS_SUBDIR)
    gt_dir = os.path.join(RAW_DATA_DIR, GT_SUBDIR)
    wind_vy_dir = os.path.join(RAW_DATA_DIR, WIND_VY_SUBDIR)
    wind_vx_dir = os.path.join(RAW_DATA_DIR, WIND_VX_SUBDIR)

    for _, row in tqdm(df_split.iterrows(), total=df_split.shape[0], desc=f"Preprocesando {split_name}"):
        sample_id, path_num = row['sample_id'], row['path_number']
        
        try:
            obstacle_map_raw = np.load(os.path.join(obstacles_dir, row['obstacle_map_file']))
            robot_path_df_raw = pd.read_csv(os.path.join(paths_dir, row['robot_path_file']))
            gt_map_raw = np.load(os.path.join(gt_dir, row['ground_truth_file']))
            wind_vy_map_raw = np.load(os.path.join(wind_vy_dir, row['wind_vy_file']))
            wind_vx_map_raw = np.load(os.path.join(wind_vx_dir, row['wind_vx_file']))

            input_X = preprocess_sample(
                obstacle_map_raw, robot_path_df_raw, wind_vy_map_raw, wind_vx_map_raw
            )
            output_Y = np.expand_dims(gt_map_raw.astype(np.float32), axis=-1)

            unique_file_id = f"{sample_id}_path_{path_num}"
            np.save(os.path.join(output_dir_X, f"{unique_file_id}_input.npy"), input_X)
            np.save(os.path.join(output_dir_Y, f"{unique_file_id}_target.npy"), output_Y)
        except Exception as e:
            print(f"ERROR procesando {sample_id}_path_{path_num}: {e}. Saltando.")
            continue

def main():
    print("--- Iniciando Preprocesamiento de Datos (5 Canales con Anemómetro Simulado) ---")
    
    cleaned_df = pd.read_csv(CLEANED_METADATA_CSV)
    train_df = cleaned_df[cleaned_df['sample_id'].isin(set(load_json(TRAIN_IDS_JSON)))]
    val_df = cleaned_df[cleaned_df['sample_id'].isin(set(load_json(VAL_IDS_JSON)))]
    test_df = cleaned_df[cleaned_df['sample_id'].isin(set(load_json(TEST_IDS_JSON)))]

    preprocess_and_save_split(train_df, "train")
    preprocess_and_save_split(val_df, "val")
    preprocess_and_save_split(test_df, "test")

    print(f"\n--- Preprocesamiento Finalizado. Datos guardados en: {PREPROCESSED_DATA_OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()