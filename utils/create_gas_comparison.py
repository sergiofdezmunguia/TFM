#!/usr/bin/env python3
"""
Script para generar comparación de mapas de gas: sin viento vs. con viento
Genera dos simulaciones idénticas excepto por la presencia de viento
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

# Agregar el directorio src/data_scripts al path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src', 'data_scripts')
sys.path.append(src_dir)

from diffusion_generator import generate_diffusion_map_roi

def create_test_environment(width=120, height=120):
    """
    Crea un entorno de prueba simple con algunos obstáculos
    """
    # Crear mapa base (valores altos = libre, valores bajos = ocupado)
    map_data = np.full((height, width), 255, dtype=np.uint8)
    
    # Añadir algunos obstáculos (rectángulos negros)
    # Obstáculo 1: pared vertical
    map_data[30:90, 20:25] = 0
    
    # Obstáculo 2: L-shape
    map_data[60:80, 40:70] = 0
    map_data[70:90, 60:70] = 0
    
    # Obstáculo 3: círculo aproximado
    center_y, center_x = 40, 80
    radius = 12
    for y in range(max(0, center_y-radius), min(height, center_y+radius+1)):
        for x in range(max(0, center_x-radius), min(width, center_x+radius+1)):
            if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                map_data[y, x] = 0
    
    return map_data

def create_gas_comparison():
    """
    Genera dos mapas de gas para comparación: sin viento vs. con viento
    """
    print("🔬 Generando comparación de mapas de gas...")
    
    # Configuración fija para ambas simulaciones
    map_data = create_test_environment()
    height, width = map_data.shape
    
    # Posición fija de la fuente de gas (área libre, evitando obstáculos)
    source_pos = (60, 30)  # Posición que sabemos funciona    # Parámetros de simulación comunes
    common_params = {
        'timesteps': 100000,  # Reducido para pruebas rápidas
        'diffusion_rate': 0.001,
        'dissipation_rate': 0.0001,
        'source_strength': 15.0,
        'verbose': True
    }
    
    print(f"📍 Fuente de gas en posición: {source_pos}")
    print(f"🗺️  Mapa de dimensiones: {height}x{width}")
    
    # Simulación 1: SIN VIENTO
    print("\n1️⃣  Simulando dispersión SIN viento...")
    result_no_wind = generate_diffusion_map_roi(
        map_subsection_np=map_data,
        source_coords_px_relative=source_pos,
        wind_source_pos=None,  # Sin viento
        wind_max_strength=0.0,
        **common_params
    )
    
    if result_no_wind[0] is None:
        print("❌ Error en simulación sin viento")
        return None
    
    gas_map_no_wind, obstacles_mask = result_no_wind[:2]
    
    # Simulación 2: CON VIENTO
    print("\n2️⃣  Simulando dispersión CON viento...")
    
    # Configuración del viento: origen en esquina superior-izquierda, sopla hacia abajo-derecha
    wind_origin = (10, 20)
    wind_direction = (1.0, 1.5)  # Vector (vy, vx) hacia abajo-derecha
    wind_strength = 0.8
    
    result_with_wind = generate_diffusion_map_roi(
        map_subsection_np=map_data,
        source_coords_px_relative=source_pos,
        wind_source_pos=wind_origin,
        wind_direction_vector=wind_direction,
        wind_max_strength=wind_strength,
        cone_angle_deg=60.0,
        wind_falloff_power=0.8,
        **common_params
    )
    
    if result_with_wind[0] is None:
        print("❌ Error en simulación con viento")
        return None
        
    gas_map_with_wind = result_with_wind[0]
    
    print("✅ Ambas simulaciones completadas exitosamente")
    
    # Crear directorio de salida
    output_dir = os.path.join(script_dir, '..', 'thesis_figures_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
        # Configurar visualización - usando el mismo estilo que las figuras existentes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Sin viento - usando estilo idéntico a las figuras existentes
    ax1.imshow(obstacles_mask, cmap='gray', interpolation='nearest')
    gas_overlay1 = ax1.imshow(gas_map_no_wind, cmap='plasma', alpha=0.8, 
                              interpolation='bilinear', vmin=0, vmax=1)
    # Solo marcar la fuente de gas
    ax1.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
                label='Fuente de Gas', zorder=10, linewidth=3)
    ax1.set_title('Dispersión de Gas SIN Viento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coordenada X (píxeles)')
    ax1.set_ylabel('Coordenada Y (píxeles)')
    ax1.legend()
    
    # Plot 2: Con viento - usando estilo idéntico a las figuras existentes
    ax2.imshow(obstacles_mask, cmap='gray', interpolation='nearest')
    gas_overlay2 = ax2.imshow(gas_map_with_wind, cmap='plasma', alpha=0.8, 
                              interpolation='bilinear', vmin=0, vmax=1)
    # Marcar fuente de gas y origen del viento
    ax2.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
                label='Fuente de Gas', zorder=10, linewidth=3)
    ax2.scatter(wind_origin[1], wind_origin[0], c='white', s=50, marker='o', 
                label='Origen Viento', zorder=10, edgecolors='black', linewidth=1)
    
    # Mostrar dirección del viento con una flecha más pequeña
    arrow_length = 8
    ax2.arrow(wind_origin[1], wind_origin[0], 
              wind_direction[1] * arrow_length, wind_direction[0] * arrow_length,
              head_width=2, head_length=2, fc='white', ec='black', alpha=0.8, linewidth=1)
    
    ax2.set_title('Dispersión de Gas CON Viento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Coordenada X (píxeles)')
    ax2.set_ylabel('Coordenada Y (píxeles)')
    ax2.legend()
    
    # Configurar el fondo negro para toda la figura
    fig.patch.set_facecolor('black')
    
    # Añadir barra de color común
    cbar = plt.colorbar(gas_overlay1, ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label('Concentración de Gas (normalizada)', rotation=270, labelpad=20)
    plt.tight_layout()
    
    # Guardar la imagen comparativa
    comparison_path = os.path.join(output_dir, 'gas_dispersion_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"💾 Imagen de comparación guardada en: {comparison_path}")
    
    # Guardar también imágenes individuales con el mismo estilo plasma
    # Imagen individual - Sin viento
    fig1, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig1.patch.set_facecolor('black')
    
    gas_display = np.copy(gas_map_no_wind)
    gas_display[obstacles_mask] = 0
    im = ax.imshow(gas_display, cmap='plasma', interpolation='bilinear', 
                   origin='upper', vmin=0, vmax=1)
    
    ax.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
               linewidth=3, zorder=10)
    ax.set_title('Dispersión de Gas sin Viento', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Coordenada X (píxeles)', color='white')
    ax.set_ylabel('Coordenada Y (píxeles)', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('black')
    
    cbar1 = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar1.set_label('Concentración de Gas (normalizada)', rotation=270, labelpad=20, color='white')
    cbar1.ax.tick_params(colors='white')
    plt.tight_layout()
    
    no_wind_path = os.path.join(output_dir, 'gas_dispersion_no_wind.png')
    plt.savefig(no_wind_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"💾 Imagen sin viento guardada en: {no_wind_path}")
    plt.close()
    
    # Imagen individual - Con viento
    fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig2.patch.set_facecolor('black')
    
    gas_display = np.copy(gas_map_with_wind)
    gas_display[obstacles_mask] = 0
    im = ax.imshow(gas_display, cmap='plasma', interpolation='bilinear', 
                   origin='upper', vmin=0, vmax=1)
    
    ax.scatter(source_pos[1], source_pos[0], c='red', s=100, marker='x', 
               linewidth=3, zorder=10)
    ax.scatter(wind_origin[1], wind_origin[0], c='white', s=80, marker='o', 
               edgecolors='black', linewidth=2, zorder=10)
    
    # Mostrar dirección del viento con una flecha
    ax.arrow(wind_origin[1], wind_origin[0], 
             wind_direction[1] * arrow_length, wind_direction[0] * arrow_length,
             head_width=3, head_length=4, fc='white', ec='black', linewidth=2, zorder=10)
    
    ax.set_title('Dispersión de Gas con Viento', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Coordenada X (píxeles)', color='white')
    ax.set_ylabel('Coordenada Y (píxeles)', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('black')
    
    cbar2 = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar2.set_label('Concentración de Gas (normalizada)', rotation=270, labelpad=20, color='white')
    cbar2.ax.tick_params(colors='white')
    plt.tight_layout()
    
    with_wind_path = os.path.join(output_dir, 'gas_dispersion_with_wind.png')
    plt.savefig(with_wind_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"💾 Imagen con viento guardada en: {with_wind_path}")
    plt.close()
    
    print("\n🎯 Comparación completada exitosamente!")
    print(f"📂 Archivos generados en: {output_dir}/")
    
    return {
        'comparison': comparison_path,
        'no_wind': no_wind_path,
        'with_wind': with_wind_path,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    random.seed(42)
    
    try:
        results = create_gas_comparison()
        if results:
            print("\n✅ Proceso completado exitosamente")
            print("📋 Archivos generados:")
            for key, path in results.items():
                if key != 'output_dir':
                    print(f"   - {key}: {os.path.basename(path)}")
        else:
            print("\n❌ Error durante la generación")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
