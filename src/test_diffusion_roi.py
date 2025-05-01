import numpy as np
import cv2
import yaml
import os
import random
import time
import math
from diffusion_generator import generate_diffusion_map_roi, get_occupied_pgm_threshold, get_free_pgm_threshold, DEFAULT_FREE_THRESH_PROB, DEFAULT_OCCUPIED_THRESH_PROB

# Parámetros de Prueba
ROI_WIDTH_PX = 256          
ROI_HEIGHT_PX = 256
MIN_FREE_SPACE_RATIO = 0.3

# Parámetros de simulación
SIM_TIMESTEPS_TEST = 1500000
SIM_DIFF_RATE_TEST = 0.0005
SIM_DISS_RATE_TEST = 0.0
SIM_SRC_STR_TEST = 10000.0
MIN_START_DISTANCE_FROM_SOURCE_PX = 50

# Rutas
MAP_NAME = "demo"
MAPS_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/maps") 
OUTPUT_TEST_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/roi_test_output") 

if __name__ == "__main__":
    print("Probando generación de UNA ROI Aleatoria")
    map_pgm_file = os.path.join(MAPS_DIR, MAP_NAME + ".pgm")
    map_yaml_file = os.path.join(MAPS_DIR, MAP_NAME + ".yaml")

    # 1. Cargar Mapa Grande 
    print(f"Cargando mapa original: {map_pgm_file}")
    try:
        with open(map_yaml_file, 'r') as f: map_metadata = yaml.safe_load(f)
        original_resolution = map_metadata['resolution']
        original_origin_x = map_metadata['origin'][0]
        original_origin_y = map_metadata['origin'][1]
        original_negate = map_metadata.get('negate', 0)

        # Calcular el umbral PGM CORRECTO para obstáculos
        yaml_free_thresh = map_metadata.get('free_thresh', DEFAULT_FREE_THRESH_PROB)
        yaml_occupied_thresh = map_metadata.get('occupied_thresh', DEFAULT_OCCUPIED_THRESH_PROB)
        OBSTACLE_PGM_MAX_VALUE = get_occupied_pgm_threshold(yaml_occupied_thresh)
        FREE_PGM_MIN_VALUE = get_free_pgm_threshold(yaml_free_thresh)

        map_image_full_np = cv2.imread(map_pgm_file, cv2.IMREAD_GRAYSCALE)
        if map_image_full_np is None: raise ValueError("No se pudo cargar PGM original")
        original_height, original_width = map_image_full_np.shape
        print(f"Mapa grande: {original_height}x{original_width}px")
    except Exception as e:
        print(f"ERROR FATAL: No se pudo cargar el mapa original: {e}"); exit(1)

    # 2. Encontrar una ROI Válida 
    roi_found = False
    roi_attempts = 0
    max_roi_attempts = 5000
    map_roi_np = None
    roi_origin_calculated = None
    rand_i_start, rand_j_start = -1, -1
    found_ratios = []

    print(f"Buscando ROI válida ({ROI_HEIGHT_PX}x{ROI_WIDTH_PX}px)...")
    while not roi_found and roi_attempts < max_roi_attempts:
        roi_attempts += 1
        rand_i_start_cand = random.randint(0, original_height - ROI_HEIGHT_PX - 1)
        rand_j_start_cand = random.randint(0, original_width - ROI_WIDTH_PX - 1)
        map_roi_np_candidate = map_image_full_np[rand_i_start_cand : rand_i_start_cand + ROI_HEIGHT_PX,
                                        rand_j_start_cand : rand_j_start_cand + ROI_WIDTH_PX]

        # Calcular espacio libre y espacio libre válido
        num_free_roi = np.sum(map_roi_np_candidate > OBSTACLE_PGM_MAX_VALUE)
        num_total_roi = ROI_WIDTH_PX * ROI_HEIGHT_PX
        free_ratio = num_free_roi / num_total_roi if num_total_roi > 0 else 0
        found_ratios.append(free_ratio)

        if free_ratio >= MIN_FREE_SPACE_RATIO:
            roi_found = True
            map_roi_np = map_roi_np_candidate
            rand_i_start = rand_i_start_cand
            rand_j_start = rand_j_start_cand

            # Calcular origen
            roi_origin_x = original_origin_x + rand_j_start * original_resolution
            roi_origin_y = original_origin_y + (original_height - (rand_i_start + ROI_HEIGHT_PX)) * original_resolution
            roi_origin_calculated = [roi_origin_x, roi_origin_y, 0.0]
            print(f"  ROI válida encontrada (Intento {roi_attempts}): Px TopLeft(i,j)=({rand_i_start},{rand_j_start}), "
                  f"Origen(x,y)=({roi_origin_x:.2f},{roi_origin_y:.2f}), Libre={free_ratio:.2f} (>= {MIN_FREE_SPACE_RATIO})")

        elif roi_attempts % 100 == 0:
             print(f"  Intento {roi_attempts}... (max ratio visto hasta ahora: {max(found_ratios):.4f})")

    if not roi_found:
        print("ERROR FATAL: No se pudo encontrar ROI válida. Ajusta parámetros (tamaño ROI, umbrales) o mapa."); exit(1)

    # 3. Encontrar Fuente Aleatoria Válida en la ROI
    source_found = False
    source_attempts = 0
    max_source_attempts = 1000
    source_pixel_relative = None

    print("Buscando fuente aleatoria válida en la ROI...")
    while not source_found and source_attempts < max_source_attempts:
        source_attempts += 1
        rand_i_rel = random.randint(0, ROI_HEIGHT_PX - 1)
        rand_j_rel = random.randint(0, ROI_WIDTH_PX - 1)

        pixel_value_roi = map_roi_np[rand_i_rel, rand_j_rel]

        if pixel_value_roi >= FREE_PGM_MIN_VALUE:
            source_found = True
            source_pixel_relative = (rand_i_rel, rand_j_rel) # Guardar los índices
            print(f"Píxel fuente válido encontrado (Intento {source_attempts}): PxRel(i,j)=({rand_i_rel},{rand_j_rel}), ValorPGM={pixel_value_roi} (> {OBSTACLE_PGM_MAX_VALUE})")
            break 

    if not source_found:
        print("ERROR FATAL: No se pudo encontrar fuente válida en la ROI."); exit(1)

    # 4. Ejecutar Simulación de Difusión en la ROI
    output_filename_test = f"roi_test_{rand_i_start}_{rand_j_start}_src_{source_pixel_relative[0]:.1f}_{source_pixel_relative[1]:.1f}.npy"
    print(f"Ejecutando simulación para la ROI seleccionada...")

    final_dense_map_roi = generate_diffusion_map_roi(
        map_subsection_np=map_roi_np,
        roi_origin=roi_origin_calculated,
        roi_resolution=original_resolution,
        roi_negate=original_negate,
        source_coords_px_relative=source_pixel_relative,
        free_thresh_prob=yaml_free_thresh,
        occupied_thresh_prob=yaml_occupied_thresh,
        timesteps=SIM_TIMESTEPS_TEST,
        diffusion_rate=SIM_DIFF_RATE_TEST,
        dissipation_rate=SIM_DISS_RATE_TEST,
        source_strength=SIM_SRC_STR_TEST,
        output_dir=OUTPUT_TEST_DIR,
        filename=output_filename_test,
        visualize_final=True,
        debug_source=True
    )

    if final_dense_map_roi is not None:
        print("\n Prueba de ROI Aleatoria Completada ")
        print(f"Archivos guardados en: {OUTPUT_TEST_DIR}")
    else:
        print("\n Prueba de ROI Aleatoria Falló ")