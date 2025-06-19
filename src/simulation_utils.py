import numpy as np
import pandas as pd
import math
import random

# Constantes y Funciones de diffusion_generator.py 
DEFAULT_FREE_THRESH_PROB = 0.196
DEFAULT_OCCUPIED_THRESH_PROB = 0.65

def get_occupied_pgm_threshold(occupied_thresh_prob=DEFAULT_OCCUPIED_THRESH_PROB):
    return int((1.0 - occupied_thresh_prob) * 255.0)

def get_free_pgm_threshold(free_thresh_prob=DEFAULT_FREE_THRESH_PROB):
    return int((1.0 - free_thresh_prob) * 255.0 + 0.999)

# Función de full_data_gen.py 
def generate_robot_path(
    obstacle_map, concentration_map, resolution,
    source_coords_px, min_distance_from_source,
    algorithm="epsilon_greedy",
    epsilon=0.2,          
    max_steps=50,
    noise_std_dev=0.01,
    free_pgm_min_value=205,
    roi_map_pgm_values=None
    ):
    height, width = obstacle_map.shape
    path_data = []

    if roi_map_pgm_values is None: return None
    if source_coords_px is None: return None

    source_i, source_j = int(round(source_coords_px[0])), int(round(source_coords_px[1]))
    
    potential_start_indices = np.argwhere(roi_map_pgm_values >= free_pgm_min_value)
    if len(potential_start_indices) == 0: return None

    valid_start_indices = []
    for idx_pair in potential_start_indices:
        start_i, start_j = idx_pair
        if obstacle_map[start_i, start_j]: continue
        distance = math.sqrt((start_i - source_i)**2 + (start_j - source_j)**2)
        if distance >= min_distance_from_source:
            valid_start_indices.append(idx_pair)

    if not valid_start_indices: return None
    
    start_idx_pair = random.choice(valid_start_indices) 
    curr_i, curr_j = start_idx_pair[0], start_idx_pair[1] 
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    for step in range(max_steps):
        try:
            concentration = concentration_map[int(round(curr_i)), int(round(curr_j))]
            if noise_std_dev > 0:
                noise = np.random.normal(0, noise_std_dev * concentration + 0.001)
                concentration = np.clip(concentration + noise, 0.0, 1.0)
        except IndexError: break 
        
        pos_x_m = curr_j * resolution + resolution / 2
        pos_y_m = curr_i * resolution + resolution / 2
        path_data.append({'step': step, 'pos_x_m': pos_x_m, 'pos_y_m': pos_y_m, 
                          'pos_i': curr_i, 'pos_j': curr_j, 'concentration': concentration})

        valid_neighbors = []
        for di, dj in possible_moves:
            ni, nj = curr_i + di, curr_j + dj
            if 0 <= ni < height and 0 <= nj < width and not obstacle_map[int(round(ni)), int(round(nj))]:
                valid_neighbors.append((ni, nj))

        if not valid_neighbors: break 

        if algorithm == "epsilon_greedy":
            if random.random() < epsilon:
                chosen_neighbor = random.choice(valid_neighbors)
            else:
                neighbor_concentrations = [concentration_map[int(round(ni)), int(round(nj))] for ni, nj in valid_neighbors]
                max_conc = max(neighbor_concentrations)
                best_indices = [idx for idx, conc in enumerate(neighbor_concentrations) if abs(conc - max_conc) < 1e-9]
                chosen_neighbor = valid_neighbors[random.choice(best_indices)]
            next_i, next_j = chosen_neighbor
        else: # random_walk o default
            next_i, next_j = random.choice(valid_neighbors)
        
        curr_i, curr_j = next_i, next_j

    if not path_data: return None
    return pd.DataFrame(path_data)