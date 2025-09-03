#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_scripts'))

def load_sample_data(sample_id, path_number, base_dir="/home/sergio/uni/master/tfm/TFM/data/gan_dataset_wind"):
    """Cargar datos de una muestra específica"""
    try:
        # Rutas correctas organizadas en subdirectorios
        gt_file = os.path.join(base_dir, "ground_truth", f"{sample_id}_gt.npy")
        obstacles_file = os.path.join(base_dir, "obstacle_maps", f"{sample_id}_obstacles.npy")
        path_file = os.path.join(base_dir, "robot_paths", f"{sample_id}_path_{path_number}.csv")
        
        if not all(os.path.exists(f) for f in [gt_file, obstacles_file, path_file]):
            missing = [f for f in [gt_file, obstacles_file, path_file] if not os.path.exists(f)]
            return None, None, None, f"Archivos no encontrados: {missing}"
        
        gt_map = np.load(gt_file)
        obstacles_map = np.load(obstacles_file)
        path_df = pd.read_csv(path_file)
        
        return gt_map, obstacles_map, path_df, None
        
    except Exception as e:
        return None, None, None, str(e)

def create_individual_example(sample_id, path_number, filename, title):
    """Crear imagen individual de un ejemplo descartado"""
    
    # Cargar datos
    gt_map, obstacles_map, path_df, error = load_sample_data(sample_id, path_number)
    
    if error:
        print(f"Error cargando {sample_id}: {error}")
        return None
    
    # Crear figura individual
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Mostrar mapa de concentración con viridis (consistente con otras figuras)
    ax.imshow(gt_map, cmap='viridis', origin='lower', interpolation='bicubic')
    
    # Crear máscara para obstáculos (negro sólido)
    obstacle_rgba = np.zeros((*obstacles_map.shape, 4))
    obstacle_rgba[obstacles_map, :] = [0, 0, 0, 1]  # Negro sólido donde hay obstáculos
    ax.imshow(obstacle_rgba, origin='lower', zorder=2)
    
    # Marcar el pico de concentración (como en figuras de difusión)
    peak_pos = np.unravel_index(np.argmax(gt_map), gt_map.shape)
    ax.scatter(peak_pos[1], peak_pos[0], c='red', s=100, marker='*', 
              edgecolors='white', linewidth=2)
    
    # Mostrar trayectoria del robot
    if not path_df.empty and all(col in path_df.columns for col in ['pos_i', 'pos_j']):
        robot_path_i = path_df['pos_i'].values
        robot_path_j = path_df['pos_j'].values
        ax.plot(robot_path_j, robot_path_i, 'lime', linewidth=2, alpha=1.0)
        
        # Marcar inicio y fin (como en figuras de difusión)
        if len(robot_path_i) > 0:
            ax.scatter(robot_path_j[0], robot_path_i[0], c='green', s=100, marker='o', 
                      edgecolors='white', linewidth=2)
            ax.scatter(robot_path_j[-1], robot_path_i[-1], c='orange', s=100, marker='s', 
                      edgecolors='white', linewidth=2)
    
    # Configurar aspecto limpio
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Guardar
    output_dir = "/home/sergio/uni/master/tfm/TFM/thesis_figures_comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Imagen guardada: {output_path}")
    plt.close()
    
    return output_path

def create_all_discarded_examples():
    """Crear las tres imágenes individuales"""
    
    examples = [
        {
            'sample_id': 'sample_00241',
            'path_number': 4,
            'filename': 'discarded_example_a',
            'title': 'sample_00241, path 4'
        },
        {
            'sample_id': 'sample_00042',
            'path_number': 0,
            'filename': 'discarded_example_b',
            'title': 'sample_00042, path 0'
        },
        {
            'sample_id': 'sample_00011',
            'path_number': 4,
            'filename': 'discarded_example_c',
            'title': 'sample_00011, path 4'
        }
    ]
    
    output_paths = []
    for example in examples:
        path = create_individual_example(
            example['sample_id'], 
            example['path_number'], 
            example['filename'], 
            example['title']
        )
        if path:
            output_paths.append(path)
    
    return output_paths

if __name__ == "__main__":
    output_paths = create_all_discarded_examples()
    if output_paths:
        print(f"\\n🎉 {len(output_paths)} imágenes individuales generadas exitosamente")
    else:
        print("\\n❌ Error al generar las imágenes")
