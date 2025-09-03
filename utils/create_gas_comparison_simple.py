#!/usr/bin/env python3
"""
Script simplificado para comparar dispersión de gas con/sin viento
Basado en la lógica existente de create_path_comparison.py pero sin robot
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import time
from tqdm import tqdm

# Agregar el directorio src/data_scripts al path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src', 'data_scripts')
sys.path.append(src_dir)

from diffusion_generator import generate_diffusion_map_roi

def load_map_data():
    """Cargar datos del mapa desde el archivo existente"""
    maps_dir = os.path.join(os.path.dirname(script_dir), 'data', 'maps')
    map_file = os.path.join(maps_dir, 'demo.pgm')
    
    if not os.path.exists(map_file):
        print(f"❌ No se encontró el archivo del mapa: {map_file}")
        return None
    
    # Leer archivo PGM manualmente
    with open(map_file, 'rb') as f:
        # Leer header
        magic = f.readline().decode('ascii').strip()
        if magic != 'P5':
            print(f"❌ Formato PGM no soportado: {magic}")
            return None
        
        # Leer dimensiones (saltando comentarios)
        line = f.readline().decode('ascii').strip()
        while line.startswith('#'):
            line = f.readline().decode('ascii').strip()
        
        width, height = map(int, line.split())
        
        # Leer valor máximo
        max_val = int(f.readline().decode('ascii').strip())
        
        # Leer datos de la imagen
        data = f.read()
        map_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    
    print(f"📍 Mapa cargado: {width}x{height} píxeles")
    
    # Reducir el tamaño del mapa para acelerar la simulación
    if width > 1000 or height > 1000:
        scale_factor = 4  # Reducir por factor de 4
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        # Hacer downsampling
        map_array_resized = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                # Tomar el mínimo de una región 4x4 (más conservador para obstáculos)
                region = map_array[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor]
                map_array_resized[i, j] = np.min(region)
        
        map_array = map_array_resized
        print(f"📍 Mapa redimensionado a: {new_width}x{new_height} píxeles para acelerar simulación")
    
    return map_array

def create_gas_comparison_simple():
    """Crear comparación simple de gas con/sin viento"""
    print("🔬 Generando comparación de dispersión de gas...")
    
    # Cargar mapa
    map_data = load_map_data()
    if map_data is None:
        return None

    # Cargar configuración YAML
    yaml_path = "/home/sergio/uni/master/tfm/TFM/data/maps/demo.yaml"
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        occupied_thresh = yaml_data.get('occupied_thresh', 0.65)
    except FileNotFoundError:
        print(f"❌ No se encontró {yaml_path}, usando umbral por defecto")
        occupied_thresh = 0.65
    
    height, width = map_data.shape
    
    # Buscar una posición libre para la fuente
    free_positions = np.where(map_data > 200)  # Píxeles libres (valores altos)
    if len(free_positions[0]) == 0:
        print("❌ No se encontraron posiciones libres en el mapa")
        return None
    
    # Seleccionar una posición cerca del centro
    center_y, center_x = height // 2, width // 2
    distances = (free_positions[0] - center_y)**2 + (free_positions[1] - center_x)**2
    closest_idx = np.argmin(distances)
    source_pos = (free_positions[0][closest_idx], free_positions[1][closest_idx])
    
    print(f"📍 Fuente de gas en: {source_pos}")
    
    # Parámetros de simulación (usando los mismos que full_data_gen.py)
    timesteps = 200000  # Aumentado ya que el mapa es más pequeño
    diffusion_rate = 0.0005
    source_strength = 1000.0
    
    # Simulación 1: SIN VIENTO
    print("1️⃣  Simulación SIN viento...")
    result_no_wind = generate_diffusion_map_roi(
        map_subsection_np=map_data,
        source_coords_px_relative=source_pos,
        occupied_thresh_prob=occupied_thresh,
        timesteps=timesteps,
        diffusion_rate=diffusion_rate,
        source_strength=source_strength,
        wind_source_pos=None,
        wind_max_strength=0.0,
        verbose=True
    )
    
    if result_no_wind[0] is None:
        print("❌ Error en simulación sin viento")
        return None
    
    gas_no_wind, obstacles = result_no_wind[:2]
    
    # Simulación 2: CON VIENTO
    print("2️⃣  Simulación CON viento...")
    wind_origin = (source_pos[0] - 30, source_pos[1] - 20)  # Arriba-izquierda de la fuente
    wind_direction = (0.7, 1.0)  # Hacia abajo-derecha
    
    result_with_wind = generate_diffusion_map_roi(
        map_subsection_np=map_data,
        source_coords_px_relative=source_pos,
        occupied_thresh_prob=occupied_thresh,
        timesteps=timesteps,
        diffusion_rate=diffusion_rate,
        source_strength=source_strength,
        wind_source_pos=wind_origin,
        wind_direction_vector=wind_direction,
        wind_max_strength=2.0,
        cone_angle_deg=60.0,
        wind_falloff_power=1.5,
        verbose=True
    )
    
    if result_with_wind[0] is None:
        print("❌ Error en simulación con viento")
        return None
    
    gas_with_wind = result_with_wind[0]
    
    # Crear visualizaciones
    output_dir = os.path.join(script_dir, '..', 'thesis_figures_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Imagen de comparación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sin viento
    ax1.imshow(obstacles, cmap='gray', interpolation='nearest')
    ax1.imshow(gas_no_wind, cmap='viridis', alpha=0.8, interpolation='bilinear')
    ax1.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
                linewidth=3, label='Fuente de Gas')
    ax1.set_title('Dispersión de Gas SIN Viento', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Con viento
    ax2.imshow(obstacles, cmap='gray', interpolation='nearest')
    im = ax2.imshow(gas_with_wind, cmap='viridis', alpha=0.8, interpolation='bilinear')
    ax2.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
                linewidth=3, label='Fuente de Gas')
    ax2.scatter(wind_origin[1], wind_origin[0], c='white', s=60, marker='o', 
                label='Origen Viento', edgecolors='black')
    
    # Flecha de viento pequeña
    ax2.arrow(wind_origin[1], wind_origin[0], 
              wind_direction[1] * 10, wind_direction[0] * 10,
              head_width=2, head_length=2, fc='white', ec='black', alpha=0.8)
    
    ax2.set_title('Dispersión de Gas CON Viento', fontsize=14, fontweight='bold')
    ax2.legend()
    
    # Colorbar
    plt.colorbar(im, ax=[ax1, ax2], shrink=0.8, 
                 label='Concentración de Gas (normalizada)')
    
    plt.tight_layout()
    
    # Guardar imágenes
    comparison_path = os.path.join(output_dir, 'gas_dispersion_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Comparación guardada: {comparison_path}")
    
    # Imágenes individuales
    for title, data, suffix in [
        ('Dispersión de Gas sin Viento', gas_no_wind, 'no_wind'),
        ('Dispersión de Gas con Viento', gas_with_wind, 'with_wind')
    ]:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(obstacles, cmap='gray', interpolation='nearest')
        im = ax.imshow(data, cmap='viridis', alpha=0.8, interpolation='bilinear')
        ax.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
                   linewidth=3, label='Fuente de Gas')
        
        if suffix == 'with_wind':
            ax.scatter(wind_origin[1], wind_origin[0], c='white', s=60, marker='o', 
                       label='Origen Viento', edgecolors='black')
            ax.arrow(wind_origin[1], wind_origin[0], 
                     wind_direction[1] * 10, wind_direction[0] * 10,
                     head_width=2, head_length=2, fc='white', ec='black', alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        plt.colorbar(im, shrink=0.8, label='Concentración de Gas (normalizada)')
        
        individual_path = os.path.join(output_dir, f'gas_dispersion_{suffix}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Imagen individual: {individual_path}")
        plt.close()
    
    print("✅ Comparación completada exitosamente!")
    return output_dir

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    try:
        output_dir = create_gas_comparison_simple()
        if output_dir:
            print(f"\n🎯 Archivos generados en: {output_dir}")
        else:
            print("\n❌ Error durante la generación")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
