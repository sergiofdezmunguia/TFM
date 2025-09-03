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
from tqdm import tqdm

from diffusion_generator import (generate_diffusion_map_roi,
                                 get_occupied_pgm_threshold,
                                 get_free_pgm_threshold,
                                 DEFAULT_FREE_THRESH_PROB,
                                 DEFAULT_OCCUPIED_THRESH_PROB)

# --- Parámetros Generales del Dataset ---
NUM_SAMPLES = 500
NUM_PATHS_PER_SAMPLE = 5
ROI_WIDTH_PX = 256; ROI_HEIGHT_PX = 256
MIN_FREE_SPACE_RATIO = 0.3

# --- Parámetros de la Simulación de Gas ---
SIM_TIMESTEPS = 1500000  
SIM_DIFF_RATE = 0.0005      
SIM_DISS_RATE = 0.0         
SIM_SRC_STR = 1000.0

# --- Parámetros de la Simulación de Viento ---
ENABLE_WIND = True 
WIND_MAX_STRENGTH_MIN = 1.0
WIND_MAX_STRENGTH_MAX = 3.0
WIND_CONE_ANGLE_MIN_DEG = 30.0
WIND_CONE_ANGLE_MAX_DEG = 90.0
WIND_FALLOFF_POWER = 1.5
WIND_SOURCE_MAX_DIST_FROM_GAS_SOURCE = 50 

# --- Parámetros de la Simulación del Robot ---
PATH_ALGORITHM = "epsilon_greedy"
EPSILON = 0.4
MAX_PATH_STEPS = 50
SENSOR_NOISE_STD_DEV = 0.01
MIN_START_DISTANCE_FROM_SOURCE_PX = 75

# --- Configuración de Rutas y Salida ---
MAP_NAME = "demo"
MAPS_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/maps")
output_suffix = "_wind" if ENABLE_WIND else "_no_wind"
OUTPUT_PARENT_DIR = os.path.expanduser(f"~/uni/master/tfm/TFM/data/gan_dataset{output_suffix}")
SAVE_VISUALIZATIONS = True

# Preparar directorios de salida
OUTPUT_GT_DIR = os.path.join(OUTPUT_PARENT_DIR, "ground_truth") 
OUTPUT_OBSTACLES_DIR = os.path.join(OUTPUT_PARENT_DIR, "obstacle_maps")
OUTPUT_PATHS_DIR = os.path.join(OUTPUT_PARENT_DIR, "robot_paths")
OUTPUT_VIS_DIR = os.path.join(OUTPUT_PARENT_DIR, "visualizations")
OUTPUT_WIND_VY_DIR = os.path.join(OUTPUT_PARENT_DIR, "wind_fields_vy")
OUTPUT_WIND_VX_DIR = os.path.join(OUTPUT_PARENT_DIR, "wind_fields_vx")
METADATA_FILE = os.path.join(OUTPUT_PARENT_DIR, "metadata.csv")

# --- Funciones de Utilidad ---
def generate_robot_path(obstacle_map, concentration_map, resolution, source_coords_px, min_distance_from_source,
                        algorithm="epsilon_greedy", epsilon=0.4, max_steps=50, noise_std_dev=0.01,
                        free_pgm_min_value=205, roi_map_pgm=None):
    height, width = obstacle_map.shape
    path_data = []
    if roi_map_pgm is None or source_coords_px is None: 
        return None
    source_i, source_j = source_coords_px
    potential_starts = np.argwhere(roi_map_pgm >= free_pgm_min_value)
    if len(potential_starts) == 0: return None
    valid_starts = [idx for idx in potential_starts if math.sqrt((idx[0]-source_i)**2 + (idx[1]-source_j)**2) >= min_distance_from_source]
    if not valid_starts: return None
    curr_i, curr_j = random.choice(valid_starts)
    moves = [(0,1),(0,-1),(1,0),(-1,0)]
    for _ in range(max_steps):
        conc = concentration_map[int(round(curr_i)), int(round(curr_j))]
        if noise_std_dev > 0: conc = np.clip(conc + np.random.normal(0, noise_std_dev), 0.0, 1.0)
        path_data.append({'pos_i':curr_i, 'pos_j':curr_j, 'concentration':conc,
                          'pos_x_m':curr_j*resolution+resolution/2, 'pos_y_m':curr_i*resolution+resolution/2})
        valid_neighbors = [(curr_i+di, curr_j+dj) for di,dj in moves if 0<=curr_i+di<height and 0<=curr_j+dj<width and not obstacle_map[curr_i+di, curr_j+dj]]
        if not valid_neighbors: continue
        if algorithm == "epsilon_greedy" and random.random() > epsilon:
            neighbor_concs = [concentration_map[ni,nj] for ni,nj in valid_neighbors]
            best_neighbor_idx = random.choice([i for i,c in enumerate(neighbor_concs) if abs(c-max(neighbor_concs))<1e-9])
            curr_i, curr_j = valid_neighbors[best_neighbor_idx]
        else:
            curr_i, curr_j = random.choice(valid_neighbors)
    return pd.DataFrame(path_data) if path_data else None

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
        obstacle_color = [0.0, 0.0, 0.0]
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
                                 label='Trayectoria (Lectura)',
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

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print(f"--- Iniciando Generación de Dataset: {'CON VIENTO' if ENABLE_WIND else 'SIN VIENTO'} ---")
    print(f"Directorio de Salida: {OUTPUT_PARENT_DIR}")

    os.makedirs(OUTPUT_GT_DIR, exist_ok=True); os.makedirs(OUTPUT_OBSTACLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PATHS_DIR, exist_ok=True); os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    if ENABLE_WIND: os.makedirs(OUTPUT_WIND_VY_DIR, exist_ok=True); os.makedirs(OUTPUT_WIND_VX_DIR, exist_ok=True)

    map_pgm_file=os.path.join(MAPS_DIR,f"{MAP_NAME}.pgm"); map_yaml_file=os.path.join(MAPS_DIR,f"{MAP_NAME}.yaml")
    with open(map_yaml_file, 'r') as f: map_metadata = yaml.safe_load(f)
    original_resolution = map_metadata['resolution']
    yaml_occ_thresh=map_metadata.get('occupied_thresh',DEFAULT_OCCUPIED_THRESH_PROB); yaml_free_thresh=map_metadata.get('free_thresh',DEFAULT_FREE_THRESH_PROB)
    FREE_PGM_MIN_VALUE = get_free_pgm_threshold(yaml_free_thresh)
    map_image_full_np = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = map_image_full_np.shape
    
    metadata_list = []
    
    for sample_idx in tqdm(range(NUM_SAMPLES), desc="Generando Escenarios"):
        sample_map_generated = False
        for _ in range(1000): # Intentos de encontrar ROI
            rand_i_start = random.randint(0, original_height - ROI_HEIGHT_PX - 1)
            rand_j_start = random.randint(0, original_width - ROI_WIDTH_PX - 1)
            map_roi_np = map_image_full_np[rand_i_start:rand_i_start+ROI_HEIGHT_PX, rand_j_start:rand_j_start+ROI_WIDTH_PX]

            if (np.sum(map_roi_np >= FREE_PGM_MIN_VALUE) / map_roi_np.size) < MIN_FREE_SPACE_RATIO: continue

            for _ in range(500): # Intentos de encontrar fuente de gas
                rand_i_rel = random.randint(0, ROI_HEIGHT_PX - 1)
                rand_j_rel = random.randint(0, ROI_WIDTH_PX - 1)
                if map_roi_np[rand_i_rel, rand_j_rel] >= FREE_PGM_MIN_VALUE:
                    source_pixel_relative = (rand_i_rel, rand_j_rel)
                    sample_map_generated = True
                    break
            if sample_map_generated: break
        
        if not sample_map_generated:
            print(f"ADVERTENCIA: No se pudo generar un escenario válido para el sample {sample_idx}. Saltando.")
            continue
            
        wind_params = {}
        if ENABLE_WIND:
            angle = random.uniform(0, 2 * math.pi); dist = random.uniform(0, WIND_SOURCE_MAX_DIST_FROM_GAS_SOURCE)
            offset_i, offset_j = dist * math.sin(angle), dist * math.cos(angle)
            wind_source_i = np.clip(source_pixel_relative[0] + offset_i, 0, ROI_HEIGHT_PX - 1)
            wind_source_j = np.clip(source_pixel_relative[1] + offset_j, 0, ROI_WIDTH_PX - 1)
            dir_angle = random.uniform(0, 2 * math.pi)
            
            wind_params = {
                "wind_source_pos": (wind_source_i, wind_source_j),
                "wind_direction_vector": (-math.sin(dir_angle), math.cos(dir_angle)),
                "wind_max_strength": random.uniform(WIND_MAX_STRENGTH_MIN, WIND_MAX_STRENGTH_MAX),
                "cone_angle_deg": random.uniform(WIND_CONE_ANGLE_MIN_DEG, WIND_CONE_ANGLE_MAX_DEG),
                "wind_falloff_power": WIND_FALLOFF_POWER
            }

        final_map, obstacle_mask, wind_vy, wind_vx = generate_diffusion_map_roi(
            map_subsection_np=map_roi_np, source_coords_px_relative=source_pixel_relative,
            occupied_thresh_prob=yaml_occ_thresh, timesteps=SIM_TIMESTEPS, diffusion_rate=SIM_DIFF_RATE,
            source_strength=SIM_SRC_STR, **wind_params
        )

        if final_map is not None and obstacle_mask is not None:
            sample_id_str = f"sample_{sample_idx:05d}"
            gt_filename = f"{sample_id_str}_gt.npy"; obs_filename = f"{sample_id_str}_obstacles.npy"
            wvy_filename = f"{sample_id_str}_wind_vy.npy"; wvx_filename = f"{sample_id_str}_wind_vx.npy"
            
            np.save(os.path.join(OUTPUT_GT_DIR, gt_filename), final_map); np.save(os.path.join(OUTPUT_OBSTACLES_DIR, obs_filename), obstacle_mask)
            if ENABLE_WIND:
                np.save(os.path.join(OUTPUT_WIND_VY_DIR, wvy_filename), wind_vy); np.save(os.path.join(OUTPUT_WIND_VX_DIR, wvx_filename), wind_vx)

            for path_num in range(NUM_PATHS_PER_SAMPLE):
                robot_path_data = generate_robot_path(obstacle_mask, final_map, original_resolution, source_pixel_relative,
                                                      MIN_START_DISTANCE_FROM_SOURCE_PX, PATH_ALGORITHM, EPSILON, 
                                                      MAX_PATH_STEPS, SENSOR_NOISE_STD_DEV, FREE_PGM_MIN_VALUE, map_roi_np)
                
                if robot_path_data is not None and not robot_path_data.empty:
                    path_filename = f"{sample_id_str}_path_{path_num}.csv"
                    robot_path_data.to_csv(os.path.join(OUTPUT_PATHS_DIR, path_filename), index=False, float_format='%.6f')
                    
                    metadata_entry = { 'sample_id': sample_id_str, 'path_number': path_num, 'map_name': MAP_NAME,
                                       'roi_origin_px_i': rand_i_start, 'roi_origin_px_j': rand_j_start,
                                       'source_relative_px_i': source_pixel_relative[0], 'source_relative_px_j': source_pixel_relative[1],
                                       'ground_truth_file': gt_filename, 'obstacle_map_file': obs_filename,
                                       'robot_path_file': path_filename, 'num_path_steps': len(robot_path_data) }
                    if ENABLE_WIND:
                        metadata_entry.update({'wind_vy_file': wvy_filename, 'wind_vx_file': wvx_filename,
                                               'wind_source_i': wind_params["wind_source_pos"][0], 'wind_source_j': wind_params["wind_source_pos"][1],
                                               'wind_dir_vy': wind_params["wind_direction_vector"][0], 'wind_dir_vx': wind_params["wind_direction_vector"][1],
                                               'wind_max_strength': wind_params["wind_max_strength"], 'wind_cone_angle_deg': wind_params["cone_angle_deg"]})
                    
                    metadata_list.append(metadata_entry)
                    if SAVE_VISUALIZATIONS:
                        vis_path = os.path.join(OUTPUT_VIS_DIR, f"{sample_id_str}_path_{path_num}.png")
                        visualize_combined(final_map, obstacle_mask, robot_path_data, source_pixel_relative, original_resolution, vis_path, title=f"Sample: {sample_id_str} Path: {path_num}")

    if metadata_list:
        pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)
    print("\n--- Generación Finalizada ---")