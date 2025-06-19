import numpy as np
import cv2
import pandas as pd
import yaml
import os
import random
import time
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from TFM.src.data_scripts.diffusion_generator import (generate_diffusion_map_roi,
                                 get_occupied_pgm_threshold,
                                 get_free_pgm_threshold,
                                 DEFAULT_FREE_THRESH_PROB,
                                 DEFAULT_OCCUPIED_THRESH_PROB)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm no instalado. Ejecuta 'pip install tqdm'", file=sys.stderr)
    def tqdm(iterable, **kwargs):
        print("INFO: tqdm no instalado, no se mostrará barra de progreso principal.")
        return iterable

# Parámetros de generación del dataset
NUM_SAMPLES = 5
ROI_WIDTH_PX = 256
ROI_HEIGHT_PX = 256
MIN_FREE_SPACE_RATIO = 0.3

# Parámetros Simulación Difusión 
SIM_TIMESTEPS = 1500000  
SIM_DIFF_RATE = 0.0005      
SIM_DISS_RATE = 0.0         
SIM_SRC_STR = 1000.0

# Parámetros Simulación Robot 
PATH_ALGORITHM = "epsilon_greedy"
EPSILON = 0.4
MAX_PATH_STEPS = 50
SENSOR_NOISE_STD_DEV = 0.01
MIN_START_DISTANCE_FROM_SOURCE_PX = 75
NUM_PATHS_PER_SAMPLE = 5

MAP_NAME = "demo"
MAPS_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/maps")

OUTPUT_PARENT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/gan_dataset-epsilon_greedy_demo")
OUTPUT_GT_DIR = os.path.join(OUTPUT_PARENT_DIR, "ground_truth") 
OUTPUT_OBSTACLES_DIR = os.path.join(OUTPUT_PARENT_DIR, "obstacle_maps")
OUTPUT_PATHS_DIR = os.path.join(OUTPUT_PARENT_DIR, "robot_paths")
OUTPUT_VIS_DIR = os.path.join(OUTPUT_PARENT_DIR, "visualizations")
METADATA_FILE = os.path.join(OUTPUT_PARENT_DIR, "metadata.csv")

SAVE_VISUALIZATIONS = True

# ==============================================
# Función de Utilidad: Generador de camino del robot
# ==============================================

def generate_robot_path(
    obstacle_map, concentration_map, resolution,
    source_coords_px, min_distance_from_source,
    algorithm="random_walk",
    epsilon=0.2,          
    max_steps=500,
    noise_std_dev=0.0, free_pgm_min_value=205,
    roi_map_pgm=None):
    """
    Genera trayectoria usando el algoritmo especificado.
    """
    height, width = obstacle_map.shape
    path_data = []

    if roi_map_pgm is None: print("ERROR PathGen: roi_map_pgm no proporcionado", file=sys.stderr); return None
    if source_coords_px is None: print("ERROR PathGen: source_coords_px no proporcionado", file=sys.stderr); return None
    source_i, source_j = source_coords_px
    potential_start_indices = np.argwhere(roi_map_pgm >= free_pgm_min_value)
    if len(potential_start_indices) == 0: print("ERROR PathGen: No se encontraron píxeles libres candidatos iniciales.", file=sys.stderr); return None

    valid_start_indices = [] # Inicializar la lista
    for idx_pair in potential_start_indices:
        start_i, start_j = idx_pair
        distance = math.sqrt((start_i - source_i)**2 + (start_j - source_j)**2)
        if distance >= min_distance_from_source:
            valid_start_indices.append(idx_pair)


    if not valid_start_indices:
        print(f"ADVERTENCIA PathGen: No se encontraron píxeles libres a la distancia mínima ({min_distance_from_source}px) de la fuente ({source_i},{source_j}).", file=sys.stderr)
        return None
    else:
        start_idx_pair = random.choice(valid_start_indices) 
        curr_i, curr_j = start_idx_pair[0], start_idx_pair[1] 

    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    for step in range(max_steps):
        # 1. Leer y Registrar Estado Actual 
        try:
            concentration = concentration_map[int(round(curr_i)), int(round(curr_j))]
            if noise_std_dev > 0:
                concentration = np.clip(concentration + np.random.normal(0, noise_std_dev), 0.0, 1.0)
        except IndexError:
            print(f"ERROR PathGen: Índice ({curr_i}, {curr_j}) fuera de límites ({height}x{width}) en paso {step}", file=sys.stderr)
            break
        pos_x_m = curr_j * resolution + resolution / 2
        pos_y_m = curr_i * resolution + resolution / 2
        path_data.append({'step': step, 'pos_x_m': pos_x_m, 'pos_y_m': pos_y_m, 'pos_i': curr_i, 'pos_j': curr_j, 'concentration': concentration})

        # 2. Calcular Siguiente Movimiento 
        next_i, next_j = curr_i, curr_j # Por defecto, quedarse quieto
        valid_neighbors = []
        for di, dj in possible_moves:
            ni, nj = curr_i + di, curr_j + dj
            if 0 <= ni < height and 0 <= nj < width and not obstacle_map[int(round(ni)), int(round(nj))]:
                valid_neighbors.append((ni, nj))

        if not valid_neighbors:
            continue

        if algorithm == "random_walk":
            chosen_neighbor = random.choice(valid_neighbors)
            next_i, next_j = chosen_neighbor

        elif algorithm == "epsilon_greedy":
            if random.random() < epsilon:
                # Exploración: vecino aleatorio válido
                chosen_neighbor = random.choice(valid_neighbors)
                next_i, next_j = chosen_neighbor
            else:
                # Explotación: mejor vecino
                neighbor_concentrations = []
                for ni, nj in valid_neighbors:
                    try:
                        neighbor_conc = concentration_map[int(round(ni)), int(round(nj))]
                        neighbor_concentrations.append(neighbor_conc)
                    except IndexError: neighbor_concentrations.append(-1.0)

                if neighbor_concentrations:
                    max_conc = max(neighbor_concentrations)
                    best_indices = [idx for idx, conc in enumerate(neighbor_concentrations) if abs(conc - max_conc) < 1e-9]
                    chosen_best_idx = random.choice(best_indices)
                    next_i, next_j = valid_neighbors[chosen_best_idx]
        else:
            print(f"ERROR PathGen: Algoritmo '{algorithm}' no reconocido.", file=sys.stderr)
            break

        # Actualizar posición
        curr_i, curr_j = next_i, next_j

    if not path_data: return None
    return pd.DataFrame(path_data)

# ==============================================
# Función de Utilidad: Visualización de Mapa
# ==============================================

def visualize_combined(gt_map, obstacle_map, robot_path_df, source_coords_px,
                        resolution, output_path, title=""):
    try:
        height, width = gt_map.shape
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10.5 * height / width)) 
        ax.set_facecolor('white')
        plot_extent = [0, width * resolution, 0, height * resolution]
        
        # Dibujar mapa de ground truth
        im_gt = ax.imshow(gt_map, cmap='viridis', origin='lower', vmin=0, vmax=1, 
                          extent=plot_extent, interpolation='nearest', alpha=0.8)

        # Dibujar obstáculos
        obstacle_rgba = np.zeros((height, width, 4), dtype=np.float32)
        obstacle_color = [0.0, 0.0, 0.0] # Negro
        obstacle_alpha = 0.7
        obstacle_rgba[obstacle_map, :3] = obstacle_color
        obstacle_rgba[obstacle_map, 3] = obstacle_alpha
        ax.imshow(obstacle_rgba, origin='lower', vmin=0, vmax=1, 
                  extent=plot_extent, interpolation='nearest', zorder=2)

        if robot_path_df is not None and not robot_path_df.empty:
            path_x, path_y, path_c = robot_path_df['pos_x_m'].values, robot_path_df['pos_y_m'].values, robot_path_df['concentration'].values
            cmap_path, norm_path = 'plasma', mcolors.Normalize(vmin=0, vmax=1)
            
            ax.plot(path_x, path_y, color='white', linestyle='-', linewidth=1.5, alpha=0.6, zorder=3)
            
            sc_path = ax.scatter(path_x, path_y, c=path_c, cmap=cmap_path, norm=norm_path, s=15, 
                                 edgecolors='black', linewidths=0.3, 
                                 label='Trayectoria (Lectura)', # Etiqueta para la leyenda principal
                                 zorder=4)
            
            cbar_path = fig.colorbar(sc_path, ax=ax, label='Lectura Norm.', 
                                     orientation='horizontal', 
                                     shrink=0.6, 
                                     aspect=25,
                                     pad=0.12,
                                     location='bottom') 
            cbar_path.ax.tick_params(labelsize=7)
            cbar_path.set_label('Lectura Norm.', fontsize=8)

            # Puntos de inicio y fin
            ax.scatter(path_x[0], path_y[0], marker='o', color='lime', s=90, edgecolors='black', 
                       label='Inicio', zorder=5)
            ax.scatter(path_x[-1], path_y[-1], marker='s', color='orange', s=90, edgecolors='black', 
                       label='Fin', zorder=5)

        # Foco GT
        src_i, src_j = source_coords_px
        source_x_m, source_y_m = src_j * resolution + resolution / 2, src_i * resolution + resolution / 2
        ax.scatter(source_x_m, source_y_m, marker='X', color='red', s=180, edgecolors='white', 
                   linewidths=1.5, label='Foco GT', zorder=6)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)

        # Leyenda principal a la derecha
        ax.legend(fontsize=8, 
                  loc='upper left', 
                  bbox_to_anchor=(1.03, 1.0),
                  borderaxespad=0.)
        
        plt.subplots_adjust(right=0.80, bottom=0.20)

        ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
        ax.set_xlim(0, width * resolution)
        ax.set_ylim(0, height * resolution)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal', adjustable='box')
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e: 
        print(f"ERROR: Fallo al crear visualización '{output_path}': {e}", file=sys.stderr)

# ==============================================================================
# SCRIPT PRINCIPAL DE GENERACIÓN DEL DATASET
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Generación de Dataset Completo")
    print(f"Número de muestras a generar: {NUM_SAMPLES}")
    print(f"Tamaño ROI: {ROI_HEIGHT_PX}x{ROI_WIDTH_PX} px")
    print(f"Parámetros Sim: Timesteps={SIM_TIMESTEPS}, DiffRate={SIM_DIFF_RATE}, SrcStr={SIM_SRC_STR}")
    print(f"Directorio Ground Truth: {OUTPUT_GT_DIR}")
    print(f"Directorio Mapas Obstáculos: {OUTPUT_OBSTACLES_DIR}")
    print(f"Directorio Trayectorias Robot: {OUTPUT_PATHS_DIR}")
    if SAVE_VISUALIZATIONS:
        print(f"Directorio Visualizaciones: {OUTPUT_VIS_DIR}")
    print(f"Archivo Metadatos: {METADATA_FILE}")
    print(f"Algoritmo Trayectoria: {PATH_ALGORITHM}, Max Pasos: {MAX_PATH_STEPS}, Ruido Sensor: {SENSOR_NOISE_STD_DEV}")

    map_pgm_file = os.path.join(MAPS_DIR, MAP_NAME + ".pgm")
    map_yaml_file = os.path.join(MAPS_DIR, MAP_NAME + ".yaml")

    # 1. Cargar Mapa Grande y Metadatos
    print(f"\nCargando mapa original: {map_pgm_file}")

    with open(map_yaml_file, 'r') as f: map_metadata = yaml.safe_load(f)
    original_resolution = map_metadata['resolution']
    original_origin_x = map_metadata['origin'][0]; original_origin_y = map_metadata['origin'][1]
    original_negate = map_metadata.get('negate', 0)
    if original_negate != 0: print("ADVERTENCIA: 'negate' no es 0 en el YAML. La interpretación de umbrales podría ser incorrecta si PGM 0 no es obstáculo.")

    # Leer umbrales del YAML o usar defaults importados
    yaml_occupied_thresh_prob = map_metadata.get('occupied_thresh', DEFAULT_OCCUPIED_THRESH_PROB)
    yaml_free_thresh_prob = map_metadata.get('free_thresh', DEFAULT_FREE_THRESH_PROB)

    # Calcular valores PGM usando funciones importadas
    OBSTACLE_PGM_MAX_VALUE = get_occupied_pgm_threshold(yaml_occupied_thresh_prob)
    FREE_PGM_MIN_VALUE = get_free_pgm_threshold(yaml_free_thresh_prob)
    print(f"  Resolución: {original_resolution:.4f} m/px | Umbral Ocupado (Prob): {yaml_occupied_thresh_prob} -> PGM <= {OBSTACLE_PGM_MAX_VALUE} | Umbral Libre (Prob): {yaml_free_thresh_prob} -> PGM >= {FREE_PGM_MIN_VALUE}")

    # Cargar imagen PGM
    map_image_full_np = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
    if map_image_full_np is None: raise ValueError(f"No se pudo cargar PGM: {map_pgm_file}")
    original_height, original_width = map_image_full_np.shape
    print(f"  Mapa base cargado: {original_height}x{original_width} px")

    # 2. Crear directorios de salida si no existen
    os.makedirs(OUTPUT_GT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OBSTACLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PATHS_DIR, exist_ok=True)
    if SAVE_VISUALIZATIONS:
        os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # Lista para guardar los metadatos
    metadata_list = []
    diffusion_maps_generated = 0
    total_paths_generated = 0
    total_roi_attempts = 0
    total_source_attempts = 0
    max_roi_attempts_per_sample = 1000
    max_source_attempts_per_roi = 500
    generation_start_time = time.time()

    # Bucle Principal de Generación
    print("\nIniciando bucle de generación de muestras...")
    pbar_samples = tqdm(range(NUM_SAMPLES), desc="Generando Mapas GT", unit="mapa")

    # Bucle principal de generación
    # ================= BUCLE EXTERNO (Mapa de Difusión) ==================
    for sample_idx in pbar_samples:
        sample_map_generated = False
        sample_roi_attempts = 0

        while not sample_map_generated and sample_roi_attempts < max_roi_attempts_per_sample:
            sample_roi_attempts += 1
            total_roi_attempts += 1

            # 3a. Seleccionar ROI Aleatoria
            if original_height <= ROI_HEIGHT_PX or original_width <= ROI_WIDTH_PX:
                print(f"ERROR FATAL: ROI ({ROI_HEIGHT_PX}x{ROI_WIDTH_PX}) > Mapa ({original_height}x{original_width})", file=sys.stderr)
                sys.exit(1)
            rand_i_start = random.randint(0, original_height - ROI_HEIGHT_PX - 1)
            rand_j_start = random.randint(0, original_width - ROI_WIDTH_PX - 1)
            map_roi_np = map_image_full_np[rand_i_start : rand_i_start + ROI_HEIGHT_PX, rand_j_start : rand_j_start + ROI_WIDTH_PX]

            # 3b. Validar ROI
            num_known_free_roi = np.sum(map_roi_np >= FREE_PGM_MIN_VALUE)
            if (num_known_free_roi / (ROI_WIDTH_PX * ROI_HEIGHT_PX)) >= MIN_FREE_SPACE_RATIO:
                sample_source_attempts = 0
                source_found_for_roi = False

                while not source_found_for_roi and sample_source_attempts < max_source_attempts_per_roi:
                    sample_source_attempts += 1
                    total_source_attempts += 1

                    # 3c. Seleccionar Fuente Aleatoria Válida 
                    rand_i_rel = random.randint(0, ROI_HEIGHT_PX - 1)
                    rand_j_rel = random.randint(0, ROI_WIDTH_PX - 1)
                    if map_roi_np[rand_i_rel, rand_j_rel] >= FREE_PGM_MIN_VALUE:
                        source_found_for_roi = True
                        source_pixel_relative = (rand_i_rel, rand_j_rel)

                        # INICIO PROCESAMIENTO MAPA GT Y OBSTÁCULOS
                        pbar_samples.set_description(f"Mapa GT {sample_idx+1}/{NUM_SAMPLES}")
                        sample_id_str = f"sample_{sample_idx:05d}"
                        gt_filename = f"{sample_id_str}_gt.npy"
                        obstacle_filename = f"{sample_id_str}_obstacles.npy"
                        gt_filepath = os.path.join(OUTPUT_GT_DIR, gt_filename)
                        obstacle_filepath = os.path.join(OUTPUT_OBSTACLES_DIR, obstacle_filename)

                        final_dense_map_normalized = None
                        obstacle_mask_roi = None

                        # 4. Ejecutar Simulación de Difusión 
                        pbar_samples.set_postfix_str("Sim. Difusión")
                        try:
                            final_dense_map_normalized, obstacle_mask_roi = generate_diffusion_map_roi(
                                map_subsection_np=map_roi_np,
                                source_coords_px_relative=source_pixel_relative,
                                occupied_thresh_prob=yaml_occupied_thresh_prob,
                                timesteps=SIM_TIMESTEPS,
                                diffusion_rate=SIM_DIFF_RATE,
                                dissipation_rate=SIM_DISS_RATE,
                                source_strength=SIM_SRC_STR,
                                verbose=False
                            )
                        except Exception as e:
                            print(f"\nERROR Inesperado llamando a generate_diffusion_map_roi para muestra {sample_idx}: {e}", file=sys.stderr)
                            final_dense_map_normalized = None

                        # 5. Procesar si la simulación fue OK 
                        if final_dense_map_normalized is not None and obstacle_mask_roi is not None:
                            map_saved_ok = False
                            
                            # 5a. Guardar Mapa de Obstáculos
                            try:
                                np.save(obstacle_filepath, obstacle_mask_roi.astype(np.uint8))
                                # 5b. Guardar Mapa Ground Truth
                                np.save(gt_filepath, final_dense_map_normalized.astype(np.float32))
                                map_saved_ok = True
                                diffusion_maps_generated += 1
                                sample_map_generated = True
                            except Exception as e:
                                print(f"\nERROR guardando GT/Obstacles para {sample_id_str}: {e}", file=sys.stderr)
                                if os.path.exists(obstacle_filepath): os.remove(obstacle_filepath)
                                if os.path.exists(gt_filepath): os.remove(gt_filepath)
                                break
                            # ======= INICIO BUCLE INTERNO (Generación de Paths) ========
                            if map_saved_ok:
                                pbar_samples.set_postfix_str("Generando Paths...")
                                paths_generated_for_this_sample = 0

                                for path_num in range(NUM_PATHS_PER_SAMPLE):
                                    pbar_samples.set_postfix_str(f"Path {path_num+1}/{NUM_PATHS_PER_SAMPLE}")
                                    path_filename = f"{sample_id_str}_path_{path_num}.csv"
                                    vis_filename = f"{sample_id_str}_path_{path_num}_visualization.png"
                                    path_filepath = os.path.join(OUTPUT_PATHS_DIR, path_filename)
                                    vis_filepath = os.path.join(OUTPUT_VIS_DIR, vis_filename)
                                    robot_path_data = None

                                    # 5c. Generar Trayectoria 
                                    try:
                                        robot_path_data = generate_robot_path(
                                            obstacle_map=obstacle_mask_roi, concentration_map=final_dense_map_normalized,
                                            resolution=original_resolution, source_coords_px=source_pixel_relative,
                                            min_distance_from_source=MIN_START_DISTANCE_FROM_SOURCE_PX,
                                            max_steps=MAX_PATH_STEPS, algorithm=PATH_ALGORITHM, epsilon=EPSILON, 
                                            noise_std_dev=SENSOR_NOISE_STD_DEV,
                                            free_pgm_min_value=FREE_PGM_MIN_VALUE, roi_map_pgm=map_roi_np
                                        )
                                    except Exception as e:
                                        print(f"\nERROR Inesperado generando Path para {sample_id_str}: {e}", file=sys.stderr)
                                        robot_path_data = None 

                                    # 5d. Guardar Trayectoria si fue exitosa
                                    if robot_path_data is not None and not robot_path_data.empty:
                                        try:
                                            robot_path_data.to_csv(path_filepath, index=False, float_format='%.6f')

                                            # 5e. Añadir a Metadatos (INCLUYE PATH_NUM)
                                            metadata_list.append({
                                                'sample_id': sample_id_str, # ID del mapa GT/Obst
                                                'path_number': path_num,   # Número de path dentro del sample
                                                'map_name': MAP_NAME,
                                                'roi_origin_px_i': rand_i_start, 'roi_origin_px_j': rand_j_start,
                                                'source_relative_px_i': rand_i_rel, 'source_relative_px_j': rand_j_rel,
                                                # Apuntan al MISMO archivo GT y Obst para todos los paths de este sample
                                                'ground_truth_file': os.path.basename(gt_filename),
                                                'obstacle_map_file': os.path.basename(obstacle_filename),
                                                'robot_path_file': os.path.basename(path_filepath), # Único para este path
                                                'num_path_steps': len(robot_path_data) })

                                            total_paths_generated += 1 # Incrementar contador total de paths
                                            paths_generated_for_this_sample += 1

                                            # 5f. Visualizar (si está activado)
                                            if SAVE_VISUALIZATIONS:
                                                visualize_combined(
                                                    gt_map=final_dense_map_normalized, obstacle_map=obstacle_mask_roi,
                                                    robot_path_df=robot_path_data, source_coords_px=source_pixel_relative,
                                                    resolution=original_resolution, output_path=vis_filepath,
                                                    title=f"Muestra: {sample_id_str} | Path: {path_num} | Fuente: ({rand_i_rel},{rand_j_rel}) | Pasos: {len(robot_path_data)}")

                                        except Exception as e:
                                            pbar_samples.write(f"\nERROR Guardando CSV/Visualizando Path {path_num} para {sample_id_str}: {e}")
                                            if os.path.exists(path_filepath):
                                                try: os.remove(path_filepath)
                                                except OSError: pass

                                    else:
                                        pbar_samples.write(f"ADVERTENCIA: No se generó Path {path_num} para {sample_id_str}. Saltando.")
                                # ======= FIN BUCLE INTERNO (Paths) ======= 
                                pbar_samples.set_postfix_str(f"{paths_generated_for_this_sample}/{NUM_PATHS_PER_SAMPLE} paths OK")

                        elif obstacle_mask_roi is not None: # Fuente en obstáculo
                             print(f"INFO: Fuente ({rand_i_rel},{rand_j_rel}) en obstáculo para ROI ({rand_i_start},{rand_j_start}). Intentando otra fuente.")

                        else: # Error interno
                            print(f"ERROR: Fallo en simulación inicial para {sample_id_str}. Descartando.", file=sys.stderr)

                        if sample_map_generated:
                            break 
            if sample_map_generated:
                break

        # Fin del bucle while ROI
        if not sample_map_generated: print(f"\nADVERTENCIA: No se pudo generar mapa GT {sample_idx} tras {max_roi_attempts_per_sample} intentos de ROI.", file=sys.stderr)
        pbar_samples.set_postfix_str("")

    # ================= FIN BUCLE EXTERNO ==================
    generation_end_time = time.time()
    pbar_samples.close()
    # 6. Guardar Metadatos 
    print("\n Guardando Archivo de Metadatos ")
    if metadata_list:
        try:
            metadata_df = pd.DataFrame(metadata_list)
            metadata_df.to_csv(METADATA_FILE, index=False); print(f"Metadatos guardados en: {METADATA_FILE}")
        except Exception as e: print(f"\nERROR CRÍTICO guardando metadatos: {e}", file=sys.stderr)
    else: print("\nADVERTENCIA: No se generaron metadatos.")

    # 7. Resumen Final 
    print("\n--- Resumen de Generación de Dataset ---")
    print(f"Tiempo total: {(generation_end_time - generation_start_time):.2f} seg")
    print(f"Mapas de Difusión (GT) generados: {diffusion_maps_generated} / {NUM_SAMPLES}")
    print(f"Trayectorias Totales generadas: {total_paths_generated} (Objetivo: {NUM_SAMPLES * NUM_PATHS_PER_SAMPLE})")
    print(f"Intentos ROI: {total_roi_attempts} | Intentos Fuente: {total_source_attempts}")
    print(f"Archivos GT en: {OUTPUT_GT_DIR}")
    print(f"Archivos Obstáculos en: {OUTPUT_OBSTACLES_DIR}")
    print(f"Archivos Trayectoria en: {OUTPUT_PATHS_DIR}")
    if SAVE_VISUALIZATIONS: print(f"Visualizaciones en: {OUTPUT_VIS_DIR}")
    if sample_map_generated > 0 and metadata_list: print(f"Metadatos en: {METADATA_FILE}")
    print(" Generación Finalizada")