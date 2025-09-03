import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm
import collections

# --- CONFIGURACIÓN DE DIRECTORIOS Y ARCHIVOS ---
OUTPUT_PARENT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/gan_dataset_wind") 
METADATA_FILE = os.path.join(OUTPUT_PARENT_DIR, "metadata.csv")
CLEANED_DATA_METADATA_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/metadata/wind_cleaned")
os.makedirs(CLEANED_DATA_METADATA_DIR, exist_ok=True)
CLEANING_LOG_FILE = os.path.join(CLEANED_DATA_METADATA_DIR, "simplified_cleaning_log.txt")

# --- UMBRALES DE LIMPIEZA ---
THRESHOLDS = {
    'min_avg_concentration_reading': 0.005, 
    'min_max_concentration_reading': 0.02,  
    'min_gt_peak_concentration': 0.05,     
    'min_reachable_free_space_ratio': 0.05, 
    'check_gt_peak_in_reachable_area': True, 
    'min_robot_reading_for_reachability_check': 0.001 
}

def get_reachable_area_from_path(obstacle_map_roi, path_coords_px_list):
    if obstacle_map_roi is None or not path_coords_px_list:
        return set(), 0
    height, width = obstacle_map_roi.shape
    queue = collections.deque()
    reachable_coords = set()
    for r_i, r_j in path_coords_px_list:
        pi, pj = int(round(r_i)), int(round(r_j))
        if 0 <= pi < height and 0 <= pj < width and not obstacle_map_roi[pi, pj]:
            if (pi, pj) not in reachable_coords:
                queue.append((pi, pj))
                reachable_coords.add((pi, pj))
    if not queue:
        return set(), 0
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        curr_i, curr_j = queue.popleft()
        for di, dj in possible_moves:
            next_i, next_j = curr_i + di, curr_j + dj
            if 0 <= next_i < height and 0 <= next_j < width and \
               not obstacle_map_roi[next_i, next_j] and (next_i, next_j) not in reachable_coords:
                reachable_coords.add((next_i, next_j))
                queue.append((next_i, next_j))
    return reachable_coords, len(reachable_coords)

def main():
    print("--- Iniciando Proceso de Limpieza de Dataset (Simplificado con Área Alcanzable) ---")
    os.makedirs(CLEANED_DATA_METADATA_DIR, exist_ok=True)
    print("\nUmbrales de limpieza a utilizar (definidos en el script):")
    for key, value in THRESHOLDS.items():
        print(f"  {key}: {value}")

    try:
        metadata_df = pd.read_csv(METADATA_FILE)
    except FileNotFoundError:
        print(f"\nERROR CRÍTICO: No se encontró el archivo de metadatos: {METADATA_FILE}", file=sys.stderr)
        sys.exit(1)
    if metadata_df.empty:
        print("\nADVERTENCIA: El archivo de metadatos está vacío.")
        sys.exit(0)

    valid_entries_list = [] 
    discard_log_list = []   

    gt_dir_full = os.path.join(OUTPUT_PARENT_DIR, "ground_truth")
    paths_dir_full = os.path.join(OUTPUT_PARENT_DIR, "robot_paths")
    obstacles_dir_full = os.path.join(OUTPUT_PARENT_DIR, "obstacle_maps")

    print(f"\nProcesando {len(metadata_df)} entradas del archivo de metadatos...")
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Limpiando datos"):
        sample_id = row.get('sample_id', f"fila_{index}") 
        path_number = row.get('path_number', 'N/A')
        gt_filename = row.get('ground_truth_file')
        path_filename = row.get('robot_path_file')
        obstacle_filename = row.get('obstacle_map_file')
        reasons_for_discard = []

        if pd.isna(gt_filename) or pd.isna(path_filename) or pd.isna(obstacle_filename):
            reasons_for_discard.append("Nombre de archivo GT, Path u Obstáculos faltante en metadatos.")
            discard_log_list.append(f"DESCARTADO (Metadatos Incompletos) - Sample: {sample_id}, Path: {path_number}, Motivos: {'; '.join(reasons_for_discard)}")
            continue

        gt_file_full_path = os.path.join(gt_dir_full, gt_filename)
        path_file_full_path = os.path.join(paths_dir_full, path_filename)
        obstacle_file_full_path = os.path.join(obstacles_dir_full, obstacle_filename)

        try:
            gt_map = np.load(gt_file_full_path)
            robot_path_df = pd.read_csv(path_file_full_path)
            obstacle_map = np.load(obstacle_file_full_path).astype(bool)
        except Exception as e:
            reasons_for_discard.append(f"Error al cargar archivos: {e}")
            discard_log_list.append(f"DESCARTADO (Carga Fallida) - Sample: {sample_id}, Path: {path_number}, Motivos: {'; '.join(reasons_for_discard)}")
            continue

        max_concentration_reading_for_this_path = 0.0
        if not robot_path_df.empty and 'concentration' in robot_path_df.columns:
            avg_concentration = robot_path_df['concentration'].mean()
            max_concentration_reading_for_this_path = robot_path_df['concentration'].max()
            if avg_concentration < THRESHOLDS['min_avg_concentration_reading']:
                reasons_for_discard.append(f"Avg_Conc_Low ({avg_concentration:.2f})")
            if max_concentration_reading_for_this_path < THRESHOLDS['min_max_concentration_reading']:
                reasons_for_discard.append(f"Max_Conc_Low ({max_concentration_reading_for_this_path:.2f})")
        elif robot_path_df.empty: reasons_for_discard.append("Path_Empty")
        elif 'concentration' not in robot_path_df.columns: reasons_for_discard.append("No_Conc_Col")

        gt_peak_value = 0.0
        gt_peak_coords_px = None
        if gt_map.size > 0:
            gt_peak_value = np.max(gt_map)
            if gt_peak_value < THRESHOLDS['min_gt_peak_concentration']:
                reasons_for_discard.append(f"GT_Peak_Low ({gt_peak_value:.2f})")
            gt_peak_coords_px = np.unravel_index(np.argmax(gt_map, axis=None), gt_map.shape)
        else: reasons_for_discard.append("GT_Empty")

        if max_concentration_reading_for_this_path >= THRESHOLDS['min_robot_reading_for_reachability_check']:
            path_coords_list = []
            if not robot_path_df.empty and {'pos_i', 'pos_j'}.issubset(robot_path_df.columns):
                path_coords_list = list(zip(robot_path_df['pos_i'], robot_path_df['pos_j']))

            if not path_coords_list:
                 reasons_for_discard.append("No_Path_Coords_For_Reach_Check")
            else:
                reachable_coords_set, num_reachable_pixels = get_reachable_area_from_path(
                    obstacle_map_roi=obstacle_map, path_coords_px_list=path_coords_list)
                total_free_pixels_in_roi = np.sum(~obstacle_map)
                if total_free_pixels_in_roi > 0:
                    reachable_ratio = num_reachable_pixels / total_free_pixels_in_roi
                    if reachable_ratio < THRESHOLDS['min_reachable_free_space_ratio']:
                        reasons_for_discard.append(f"Reach_Ratio_Low ({reachable_ratio*100:.1f}%)")
                else: reasons_for_discard.append("ROI_No_Free_Space")
                if THRESHOLDS['check_gt_peak_in_reachable_area'] and gt_peak_coords_px is not None:
                    if gt_peak_coords_px not in reachable_coords_set:
                        reasons_for_discard.append("GT_Peak_Not_Reachable")
        
        if not reasons_for_discard:
            valid_entries_list.append(row.to_dict()) 
        else:
            discard_log_list.append(f"DESCARTADO - S:{sample_id}, P:{path_number}, Motivos: {'; '.join(reasons_for_discard)}")

    try:
        with open(CLEANING_LOG_FILE, 'w') as f:
            for entry in discard_log_list: f.write(entry + "\n")
        print(f"\nLog de limpieza ({len(discard_log_list)} desc.) guardado en: {CLEANING_LOG_FILE}")
    except Exception as e: print(f"\nERROR guardando log: {e}")

    valid_metadata_df = pd.DataFrame(valid_entries_list)
    if not valid_metadata_df.empty:
        cleaned_metadata_csv_path = os.path.join(CLEANED_DATA_METADATA_DIR, "cleaned_metadata.csv")
        try:
            valid_metadata_df.to_csv(cleaned_metadata_csv_path, index=False)
            print(f"Metadatos válidos ({len(valid_metadata_df)}) guardados en: {cleaned_metadata_csv_path}")
        except Exception as e: print(f"\nERROR guardando cleaned_metadata.csv: {e}"); sys.exit(1)

        if 'sample_id' not in valid_metadata_df.columns:
            print("\nADVERTENCIA: Columna 'sample_id' no encontrada. No se puede dividir train/val/test.")
        else:
            unique_sample_ids = valid_metadata_df['sample_id'].unique()
            np.random.seed(42); np.random.shuffle(unique_sample_ids) 
            num_unique_samples = len(unique_sample_ids)
            train_ratio, val_ratio = 0.7, 0.15 
            train_split_idx = int(train_ratio * num_unique_samples)
            val_split_idx = int((train_ratio + val_ratio) * num_unique_samples)
            train_sample_ids = unique_sample_ids[:train_split_idx]
            val_sample_ids = unique_sample_ids[train_split_idx:val_split_idx]
            test_sample_ids = unique_sample_ids[val_split_idx:]
            try:
                for split_name, ids_list in [("train", train_sample_ids), ("val", val_sample_ids), ("test", test_sample_ids)]:
                    with open(os.path.join(CLEANED_DATA_METADATA_DIR, f'{split_name}_sample_ids.json'), 'w') as f:
                        json.dump(ids_list.tolist(), f, indent=4)
                print(f"\nDivisión de IDs de muestras: Train:{len(train_sample_ids)}, Val:{len(val_sample_ids)}, Test:{len(test_sample_ids)}")
                train_routes_count = valid_metadata_df[valid_metadata_df['sample_id'].isin(train_sample_ids)].shape[0]
                val_routes_count = valid_metadata_df[valid_metadata_df['sample_id'].isin(val_sample_ids)].shape[0]
                test_routes_count = valid_metadata_df[valid_metadata_df['sample_id'].isin(test_sample_ids)].shape[0]
                print(f"Rutas finales: Train:{train_routes_count}, Val:{val_routes_count}, Test:{test_routes_count}")
            except Exception as e: print(f"\nERROR guardando JSONs de división: {e}")
    else: print("\nADVERTENCIA: Ninguna entrada superó los filtros.")
    print("\n--- Proceso de Limpieza de Dataset Finalizado ---")

if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_PARENT_DIR):
        print(f"ERROR FATAL: OUTPUT_PARENT_DIR '{OUTPUT_PARENT_DIR}' no existe.")
        sys.exit(1)
    if not os.path.exists(METADATA_FILE):
        print(f"ERROR FATAL: METADATA_FILE '{METADATA_FILE}' no existe.")
        sys.exit(1)
    main()