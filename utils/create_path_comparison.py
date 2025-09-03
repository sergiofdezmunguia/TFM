import os
import site

# CONFIGURAR CUDA LIBRARIES ANTES DE CUALQUIER IMPORT
# Add CUDA library paths before importing any CuPy-dependent modules
cuda_paths = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12/lib64", 
    "/usr/local/cuda-11/lib64",
    "/opt/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/wsl/lib"
]

# Also add Python site-packages CUDA libraries
try:
    site_packages = site.getsitepackages()
    for site_dir in site_packages:
        nvidia_dirs = [
            os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib"),
            os.path.join(site_dir, "nvidia", "cuda_runtime", "lib"),
            os.path.join(site_dir, "nvidia", "cublas", "lib"),
            os.path.join(site_dir, "nvidia", "cusolver", "lib"),
            os.path.join(site_dir, "nvidia", "curand", "lib"),
        ]
        cuda_paths.extend([d for d in nvidia_dirs if os.path.exists(d)])
except:
    pass

# Set LD_LIBRARY_PATH to include CUDA paths
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
cuda_path_str = ':'.join(cuda_paths)
os.environ['LD_LIBRARY_PATH'] = f"{cuda_path_str}:{current_ld_path}" if current_ld_path else cuda_path_str

import numpy as np
import cv2
import pandas as pd
import yaml
import random
import time
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

print("🚀 INICIANDO SCRIPT CREATE_PATH_COMPARISON...")

# Add the data_scripts directory to Python path for importing diffusion_generator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_scripts'))

print("📥 Importando diffusion_generator...")
from diffusion_generator import (generate_diffusion_map_roi,
                                 get_occupied_pgm_threshold,
                                 get_free_pgm_threshold,
                                 DEFAULT_FREE_THRESH_PROB,
                                 DEFAULT_OCCUPIED_THRESH_PROB)
print("✅ Import de diffusion_generator exitoso")

# --- PANEL DE CONTROL PARA GENERAR FIGURAS ---
# --- Cambia los valores aquí para generar las imágenes que necesitas ---

# 1. Algoritmos de ruta a generar (ambos en el mismo escenario)
PATH_ALGORITHMS_TO_GENERATE = ["epsilon_greedy", "random_walk"]  # Se generarán ambos

# 2. Elige una semilla para generar un escenario reproducible.
#    Cambia este número para obtener diferentes mapas y posiciones de fuentes.
RANDOM_SEED = 123 

# 3. Directorio de salida para las figuras
FIGURES_OUTPUT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/thesis_figures")

# --- PARÁMETROS FIJOS PARA LA FIGURA ---
# (Estos son tomados de tu script original)
ROI_WIDTH_PX = 256; ROI_HEIGHT_PX = 256
MIN_FREE_SPACE_RATIO = 0.3
SIM_TIMESTEPS = 1500000; SIM_DIFF_RATE = 0.0005; SIM_SRC_STR = 1000.0
ENABLE_WIND = False # Desactivamos el viento para una comparación clara de algoritmos
EPSILON = 0.4; MAX_PATH_STEPS = 50; SENSOR_NOISE_STD_DEV = 0.01
MIN_START_DISTANCE_FROM_SOURCE_PX = 75
MAP_NAME = "demo"
MAPS_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/maps")

# --- NO ES NECESARIO EDITAR DEBAJO DE ESTA LÍNEA ---
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# --- FUNCIONES DE UTILIDAD ---
# Tu función generate_robot_path original
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

# NUEVA función de visualización, mucho más simple y limpia
def visualize_path_comparison(gt_map, obstacle_map, robot_path_df, source_coords_px,
                              resolution, output_path, title=""):
    try:
        height, width = gt_map.shape
        # Crear figura sin márgenes ni ejes
        fig = plt.figure(figsize=(10, 10 * height / width), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plot_extent = [0, width, 0, height] # Trabajar en coordenadas de píxeles

        # 1. Dibujar el heatmap de gas
        ax.imshow(gt_map, cmap='viridis', origin='lower', extent=plot_extent, interpolation='bicubic')

        # 2. Dibujar los obstáculos
        obstacle_rgba = np.zeros((height, width, 4)); 
        obstacle_rgba[obstacle_map, :3] = 0.1 # Gris oscuro
        obstacle_rgba[obstacle_map, 3] = 0.8
        ax.imshow(obstacle_rgba, origin='lower', extent=plot_extent, zorder=2)
        
        # 3. Dibujar la ruta
        if robot_path_df is not None and not robot_path_df.empty:
            path_x_px = robot_path_df['pos_j'].values
            path_y_px = robot_path_df['pos_i'].values
            
            # Línea principal de la trayectoria
            ax.plot(path_x_px, path_y_px, color='white', linestyle='-', linewidth=2, alpha=0.8, zorder=3)
            
            # Punto de inicio (Verde) y fin (Naranja)
            ax.scatter(path_x_px[0], path_y_px[0], marker='o', color='lime', s=150, edgecolors='black', zorder=5)
            ax.scatter(path_x_px[-1], path_y_px[-1], marker='s', color='orange', s=150, edgecolors='black', zorder=5)

        # 4. Dibujar la fuente de gas
        src_i, src_j = source_coords_px
        ax.scatter(src_j, src_i, marker='X', color='red', s=250, edgecolors='white', linewidths=1.5, zorder=6)

        # 5. Añadir un título simple y limpio
        ax.text(width / 2, height - 15, title, color='white', ha='center', va='center',
                fontsize=20, bbox=dict(facecolor='black', alpha=0.5))

        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e: 
        print(f"ERROR: Fallo al crear visualización '{output_path}': {e}", file=sys.stderr)

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("Iniciando script...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print(f"Generando figuras para algoritmos: {PATH_ALGORITHMS_TO_GENERATE} con semilla {RANDOM_SEED}")

    # Cargar mapa y metadatos
    print("Cargando mapa...")
    map_pgm_file = os.path.join(MAPS_DIR, f"{MAP_NAME}.pgm")
    map_yaml_file = os.path.join(MAPS_DIR, f"{MAP_NAME}.yaml")
    print(f"Archivo PGM: {map_pgm_file}")
    print(f"Archivo YAML: {map_yaml_file}")
    
    if not os.path.exists(map_pgm_file):
        print(f"ERROR: No se encontró el archivo {map_pgm_file}")
        sys.exit(1)
    if not os.path.exists(map_yaml_file):
        print(f"ERROR: No se encontró el archivo {map_yaml_file}")
        sys.exit(1)
        
    with open(map_yaml_file, 'r') as f: map_metadata = yaml.safe_load(f)
    original_resolution = map_metadata['resolution']
    free_pgm_min_value = get_free_pgm_threshold(map_metadata.get('free_thresh', DEFAULT_FREE_THRESH_PROB))
    occ_thresh_prob = map_metadata.get('occupied_thresh', DEFAULT_OCCUPIED_THRESH_PROB)
    map_image_full_np = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = map_image_full_np.shape

    # Generar un escenario determinista gracias a la semilla
    sample_map_generated = False
    for _ in range(1000): # Intentar encontrar un escenario válido
        rand_i_start = random.randint(0, original_height - ROI_HEIGHT_PX - 1)
        rand_j_start = random.randint(0, original_width - ROI_WIDTH_PX - 1)
        map_roi_np = map_image_full_np[rand_i_start:rand_i_start+ROI_HEIGHT_PX, rand_j_start:rand_j_start+ROI_WIDTH_PX]
        if (np.sum(map_roi_np >= free_pgm_min_value) / map_roi_np.size) < MIN_FREE_SPACE_RATIO: continue
        for _ in range(500):
            rand_i_rel, rand_j_rel = random.randint(0, ROI_HEIGHT_PX-1), random.randint(0, ROI_WIDTH_PX-1)
            if map_roi_np[rand_i_rel, rand_j_rel] >= free_pgm_min_value:
                source_pixel_relative = (rand_i_rel, rand_j_rel); sample_map_generated = True; break
        if sample_map_generated: break
    
    if not sample_map_generated:
        print("ERROR: No se pudo generar escenario. Prueba otra semilla."); sys.exit(1)
        
    # Simular el entorno (siempre será el mismo para la misma semilla)
    print("Simulando difusión de gas...")
    final_map, obstacle_mask = generate_diffusion_map_roi(
        map_subsection_np=map_roi_np, source_coords_px_relative=source_pixel_relative,
        occupied_thresh_prob=occ_thresh_prob, timesteps=SIM_TIMESTEPS, 
        diffusion_rate=SIM_DIFF_RATE, source_strength=SIM_SRC_STR, wind_max_strength=0.0
    )

    # Generar las rutas para cada algoritmo usando el mismo entorno
    for algorithm in PATH_ALGORITHMS_TO_GENERATE:
        print(f"\n--- Generando ruta con algoritmo: {algorithm} ---")
        
        # Resetear semilla para generar rutas diferentes pero reproducibles
        algorithm_seed = RANDOM_SEED + hash(algorithm) % 10000
        random.seed(algorithm_seed)
        np.random.seed(algorithm_seed)
        
        # Generar la ruta del robot
        robot_path_data = generate_robot_path(
            obstacle_map=obstacle_mask, concentration_map=final_map, resolution=original_resolution,
            source_coords_px=source_pixel_relative, min_distance_from_source=MIN_START_DISTANCE_FROM_SOURCE_PX,
            algorithm=algorithm, epsilon=EPSILON, max_steps=MAX_PATH_STEPS,
            noise_std_dev=SENSOR_NOISE_STD_DEV, free_pgm_min_value=free_pgm_min_value, roi_map_pgm=map_roi_np
        )
        
        if robot_path_data is None:
            print(f"ERROR: No se pudo generar la ruta para {algorithm}"); continue
            
        # Visualizar y guardar la figura limpia
        output_filename = f"path_comparison_{algorithm}.png"
        output_path = os.path.join(FIGURES_OUTPUT_DIR, output_filename)
        title = f"Estrategia: {algorithm.replace('_', ' ').title()}"
        
        visualize_path_comparison(
            gt_map=final_map, obstacle_map=obstacle_mask, robot_path_df=robot_path_data,
            source_coords_px=source_pixel_relative, resolution=original_resolution,
            output_path=output_path, title=title
        )
        
        print(f"Figura guardada en: {output_path}")

    print(f"\n🎉 ¡Completado! Se generaron {len(PATH_ALGORITHMS_TO_GENERATE)} figuras en {FIGURES_OUTPUT_DIR}")