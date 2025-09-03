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

def create_discarded_examples_figure():
    """Crear figura con ejemplos de muestras descartadas usando formato de subfiguras"""
    
    # Ejemplos seleccionados del log de limpieza - casos más representativos
    examples = [
        {
            'sample_id': 'sample_00241',
            'path_number': 4,
            'reason': 'pico de concentración no alcanzable por el robot',
            'description': 'sample_00241, path 4',
            'label': 'FIG:DISCARDED_A'
        },
        {
            'sample_id': 'sample_00042', 
            'path_number': 0,
            'reason': 'fuente de gas aislada por obstáculos',
            'description': 'sample_00042, path 0',
            'label': 'FIG:DISCARDED_B'
        },
        {
            'sample_id': 'sample_00011',
            'path_number': 4, 
            'reason': 'área explorable insuficiente (3.4%)',
            'description': 'sample_00011, path 4',
            'label': 'FIG:DISCARDED_C'
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, example in enumerate(examples):
        ax = axes[i]
        
        # Cargar datos
        gt_map, obstacles_map, path_df, error = load_sample_data(
            example['sample_id'], 
            example['path_number']
        )
        
        if error:
            ax.text(0.5, 0.5, f"Error: {error}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{example['sample_id']}, Path {example['path_number']}\n{example['description']}")
            continue
            
        # Visualizar usando el mismo diseño que las predicciones del modelo
        # Fondo azul y obstáculos negros como en la imagen de referencia
        
        # Crear imagen base con fondo azul
        base_image = np.ones_like(obstacles_map, dtype=float)
        
        # Obstáculos en negro (valor 0)
        base_image[obstacles_map] = 0.0
        
        # Mostrar imagen base con fondo azul
        ax.imshow(base_image, cmap='Blues', vmin=0, vmax=1, alpha=1.0)
        
                # Superponer mapa de concentración con colormap viridis (amarillo a verde/azul claro)
        if gt_map is not None:
            gt_masked = np.ma.masked_where(gt_map < 1e-6, gt_map)
            im = ax.imshow(gt_masked, cmap='viridis', alpha=0.8, vmin=0, vmax=np.max(gt_map))
            
            # Encontrar y marcar el pico de concentración con estrella roja (como en figuras de difusión)
            peak_pos = np.unravel_index(np.argmax(gt_map), gt_map.shape)
            ax.scatter(peak_pos[1], peak_pos[0], c='red', s=100, marker='*', 
                      edgecolors='white', linewidth=2)
        
        # Mostrar trayectoria del robot en verde lima
        if not path_df.empty and all(col in path_df.columns for col in ['pos_i', 'pos_j']):
            robot_path_i = path_df['pos_i'].values
            robot_path_j = path_df['pos_j'].values
            ax.plot(robot_path_j, robot_path_i, 'lime', linewidth=2, alpha=1.0)
            
            # Marcar inicio y fin como en las figuras de difusión
            if len(robot_path_i) > 0:
                ax.scatter(robot_path_j[0], robot_path_i[0], c='green', s=100, marker='o', 
                          edgecolors='white', linewidth=2)
                ax.scatter(robot_path_j[-1], robot_path_i[-1], c='orange', s=100, marker='s', 
                          edgecolors='white', linewidth=2)
        
        # Remover estadísticas del texto superpuesto
        
        # Título limpio solo con el ID de la muestra
        ax.set_title(f"{example['description']}", fontsize=12, fontweight='bold')
        
        # Sin subtítulos en xlabel para mejor legibilidad
        ax.set_xlabel('Píxeles', fontsize=10)
        ax.set_ylabel('Píxeles', fontsize=10)
        
        # Remover ticks para un aspecto más limpio
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Sin leyenda para mantener limpio
    
    plt.tight_layout()
    
    # Guardar
    output_dir = "/home/sergio/uni/master/tfm/TFM/thesis_figures_comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "discarded_examples.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Ejemplos de muestras descartadas guardados en: {output_path}")
    
    plt.close()
    
    return output_path

if __name__ == "__main__":
    output_path = create_discarded_examples_figure()
    if output_path:
        print(f"\\n🎉 Figura de ejemplos descartados generada: {output_path}")
    else:
        print("\\n❌ Error al generar la figura")
