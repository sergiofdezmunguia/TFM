import numpy as np
import time
import sys

try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False
    def tqdm(iterable, **kwargs):
        print("INFO: tqdm no instalado, no se mostrará barra de progreso.")
        return iterable

import os
import ctypes
import site

# Add CUDA library paths before importing CuPy
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

try:
    import cupy as cp
    if not cp.is_available(): 
        raise ImportError("CuPy instalado pero no detecta GPU CUDA.")
    _use_gpu = True
    print("INFO: Usando CuPy (GPU) para la simulación.")
except (ImportError, RuntimeError, Exception) as e:
    import numpy as np
    cp = np 
    _use_gpu = False
    print(f"INFO: CuPy no disponible ({e}), usando NumPy (CPU) para la simulación.")

# Parámetros por Defecto
DEFAULT_TIMESTEPS = 750000
DEFAULT_DIFFUSION_RATE = 0.0005
DEFAULT_DISSIPATION_RATE = 0.0
DEFAULT_SOURCE_STRENGTH = 10.0

# Valores de map_server
DEFAULT_FREE_THRESH_PROB = 0.196
DEFAULT_OCCUPIED_THRESH_PROB = 0.65

def get_occupied_pgm_threshold(occupied_thresh_prob=DEFAULT_OCCUPIED_THRESH_PROB):
    """Calcula el valor PGM (0-255) MÁXIMO que se considera obstáculo."""
    return int((1.0 - occupied_thresh_prob) * 255.0)

def get_free_pgm_threshold(free_thresh_prob=DEFAULT_FREE_THRESH_PROB):
    """Calcula el valor PGM (0-255) MÍNIMO que se considera libre."""
    return int((1.0 - free_thresh_prob) * 255.0 + 0.999)

def create_localized_wind_field(height, width, 
                                wind_source_pos, 
                                wind_direction_vec,
                                wind_max_strength,
                                cone_angle_deg,
                                falloff_power=1.0):
    """
    Crea un campo de vectores de viento 2D (vy, vx) que se origina en un punto
    y se propaga en forma de cono, atenuándose con la distancia
    """
    xp = cp if _use_gpu else np
    
    y_coords, x_coords = xp.indices((height, width), dtype=xp.float32)
    
    source_y, source_x = wind_source_pos
    vecs_x = x_coords - source_x
    vecs_y = y_coords - source_y
    
    distances = xp.sqrt(vecs_x**2 + vecs_y**2)
    distances[distances < 1e-6] = 1e-6

    # Vectores de dirección normalizados desde la fuente a cada píxel
    dir_vecs_x = vecs_x / distances
    dir_vecs_y = vecs_y / distances
    
    # Normalizar el vector de dirección principal del viento
    main_dir_y, main_dir_x = wind_direction_vec
    main_dir_norm = xp.sqrt(main_dir_x**2 + main_dir_y**2)
    if float(main_dir_norm) == 0: # Si no hay dirección, no hay viento
        return xp.zeros((height, width), dtype=xp.float32), xp.zeros((height, width), dtype=xp.float32)
    
    main_dir_x_norm = main_dir_x / main_dir_norm
    main_dir_y_norm = main_dir_y / main_dir_norm
    
    # Calcular producto escalar para determinar qué píxeles están en el cono
    dot_product = dir_vecs_x * main_dir_x_norm + dir_vecs_y * main_dir_y_norm
    
    # Crear máscara para el cono de viento
    cone_angle_rad = xp.deg2rad(cone_angle_deg)
    cone_mask = dot_product > xp.cos(cone_angle_rad / 2)
    
    # Calcular atenuación de la fuerza con la distancia
    strength_falloff = wind_max_strength / (distances**falloff_power)
    
    # Aplicar máscara y atenuación al campo de viento
    # El viento fluye desde la fuente hacia afuera, en la dirección de los vectores dir_vecs
    final_vx = dir_vecs_x * strength_falloff * cone_mask
    final_vy = dir_vecs_y * strength_falloff * cone_mask
    
    return final_vy, final_vx

def generate_diffusion_map_roi(
    map_subsection_np, 
    source_coords_px_relative,
    occupied_thresh_prob = DEFAULT_OCCUPIED_THRESH_PROB,
    timesteps=DEFAULT_TIMESTEPS,
    diffusion_rate=DEFAULT_DIFFUSION_RATE,
    dissipation_rate=DEFAULT_DISSIPATION_RATE,
    source_strength=DEFAULT_SOURCE_STRENGTH,
    wind_source_pos=None,         # Tupla (y, x) del origen del viento
    wind_direction_vector=(0,0),  # Tupla (vy, vx) de la dirección principal
    wind_max_strength=0.0,        # Fuerza máxima en el origen
    cone_angle_deg=45.0,          # Ángulo de apertura del cono
    wind_falloff_power=1.0,       # Atenuación con la distancia
    verbose=False
    ):
    """
    Genera mapa de concentración por difusión Laplaciana en una ROI.
    NO GUARDA ARCHIVOS.

    Args:
        map_subsection_np (np.ndarray): Array 2D de la ROI del mapa PGM (uint8).
        source_coords_px_relative (tuple): Coordenadas (i, j) del píxel fuente relativas a la ROI.
        occupied_thresh_prob (float): Probabilidad mínima para considerar un píxel como obstáculo.
        timesteps (int): Número de pasos de la simulación.
        diffusion_rate (float): Tasa de difusión.
        dissipation_rate (float): Tasa de disipación (0 a 1).
        source_strength (float): Fuerza/valor inicial en el píxel fuente.
        wind_source_pos (tuple | None): Coordenadas (i, j) del origen del viento. Si es None, no hay viento.
        wind_direction_vector (tuple): Vector (vy, vx) de la dirección principal del viento.
        wind_max_strength (float): Fuerza máxima del viento en su origen.
        cone_angle_deg (float): Ángulo de apertura del cono de viento en grados.
        wind_falloff_power (float): Potencia de atenuación de la fuerza del viento con la distancia.
        verbose (bool): Si imprimir mensajes de progreso/debug internos.

    Returns:
        tuple: (final_map_np, obstacle_mask, wind_vy, wind_vx)
               - final_map_np (np.ndarray | None): Mapa de concentración final normalizado (0-1) en CPU (float32), o None si hay error.
               - obstacle_mask (np.ndarray | None): Máscara booleana de obstáculos (True=Obstáculo) en CPU, o None si hay error.
               - wind_vy (np.ndarray | None): Componente vertical del viento en CPU, o None si hay error.
               - wind_vx (np.ndarray | None): Componente horizontal del viento en CPU, o None si hay error.
               Devuelve (None, None, None, None) si la fuente está en un obstáculo o error inicial.
    """
    global cp, _use_gpu
    xp = cp if _use_gpu else np
    start_time = time.time()

    # 1. Validar y Preparar Datos de Entrada
    try:
        if not isinstance(map_subsection_np, np.ndarray) or map_subsection_np.ndim != 2:
            raise ValueError("map_subsection_np debe ser un array 2D de NumPy.")
        height, width = map_subsection_np.shape
        if height < 3 or width < 3:
             raise ValueError(f"ROI muy pequeña ({height}x{width}px), se necesitan al menos 3x3.")

        if not (isinstance(source_coords_px_relative, tuple) and len(source_coords_px_relative) == 2):
             raise ValueError("source_coords_px_relative debe ser una tupla (i, j).")
        source_i, source_j = map(int, source_coords_px_relative)

        if not (0 <= source_i < height and 0 <= source_j < width):
             raise ValueError(f"Coordenadas fuente ({source_i},{source_j}) fuera de los límites de la ROI {height}x{width}.")

    except Exception as e:
        print(f"ERROR [DiffusionSim]: Procesando datos de entrada: {e}", file=sys.stderr)
        return None, None, None, None

    # 2. Calcular Umbral y Crear Máscara de Obstáculos
    OBSTACLE_PGM_MAX_VALUE = get_occupied_pgm_threshold(occupied_thresh_prob)
    obstacle_mask_xp = xp.asarray(map_subsection_np <= OBSTACLE_PGM_MAX_VALUE)

    # 3. Validar Píxel Fuente 
    is_source_obstacle = obstacle_mask_xp[source_i, source_j]
    if isinstance(is_source_obstacle, cp.ndarray):
        is_source_obstacle = bool(is_source_obstacle.get())
    else:
        is_source_obstacle = bool(is_source_obstacle)

    if is_source_obstacle:
        if verbose: print(f"ADVERTENCIA [DiffusionSim]: Píxel fuente ({source_i},{source_j}) está en un obstáculo (Valor PGM {map_subsection_np[source_i, source_j]} <= {OBSTACLE_PGM_MAX_VALUE}). No se ejecutará simulación.")
        obstacle_mask_cpu = cp.asnumpy(obstacle_mask_xp) if _use_gpu else obstacle_mask_xp
        return None, obstacle_mask_cpu.astype(bool), None, None

    # 4. Inicializar Grid de Concentración
    concentration_grid = xp.zeros((height, width), dtype=xp.float32)
    # Añadir fuente inicial solo si no es obstáculo
    concentration_grid[source_i, source_j] = xp.float32(source_strength)
    if verbose: print(f"  [DiffusionSim] DEBUG: Added initial source strength {source_strength} at ({source_i},{source_j})")
    
    # Pre-calcular el campo de viento
    wind_field_vy, wind_field_vx = xp.zeros_like(concentration_grid), xp.zeros_like(concentration_grid)
    if wind_source_pos is not None and wind_max_strength > 0:
        wind_field_vy, wind_field_vx = create_localized_wind_field(
            height, width, wind_source_pos, wind_direction_vector, 
            wind_max_strength, cone_angle_deg, wind_falloff_power)
        
    # 5. Bucle Principal de Simulación
    loop_iterator = range(timesteps)
    if verbose:
        try:
            from tqdm import tqdm as internal_tqdm
            loop_iterator = internal_tqdm(loop_iterator, desc="Simulación Difusión Interna", leave=False, disable=False)
        except ImportError:
            pass

    for _ in loop_iterator:
        current_grid = xp.copy(concentration_grid)

        # a) Calcular Laplaciano
        laplacian = (current_grid[:-2, 1:-1] + current_grid[2:, 1:-1] +  # Arriba, Abajo
                     current_grid[1:-1, :-2] + current_grid[1:-1, 2:] -  # Izquierda, Derecha
                     4 * current_grid[1:-1, 1:-1])                       # Centro * 4
        change_diffusion = diffusion_rate * laplacian

        # b) Calcular Advección (Viento)
        grad_x = xp.zeros_like(current_grid)
        grad_y = xp.zeros_like(current_grid)
        # Usar diferencias centrales para gradiente, más simple
        grad_x[:, 1:-1] = (current_grid[:, 2:] - current_grid[:, :-2]) / 2.0
        grad_y[1:-1, :] = (current_grid[2:, :] - current_grid[:-2, :]) / 2.0
        change_advection = - (wind_field_vx[1:-1, 1:-1] * grad_x[1:-1, 1:-1] + 
                              wind_field_vy[1:-1, 1:-1] * grad_y[1:-1, 1:-1])

        # c) Actualizar concentración
        non_obstacle_interior_mask = ~obstacle_mask_xp[1:-1, 1:-1]
        total_change = change_diffusion + change_advection
        concentration_grid[1:-1, 1:-1][non_obstacle_interior_mask] += total_change[non_obstacle_interior_mask]

        # d) Disipación y condiciones de contorno
        if dissipation_rate > 1e-9:
            concentration_grid *= (1.0 - dissipation_rate)

        concentration_grid[obstacle_mask_xp] = 0.0
        xp.maximum(concentration_grid, 0.0, out=concentration_grid)

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"  [DiffusionSim] Simulación completada en {elapsed_time:.2f} segundos.")

    # 6. Normalizar Resultado 
    final_map = concentration_grid
    max_val = xp.max(final_map)

    if verbose: print(f"  [DiffusionSim] DEBUG: Max concentración ANTES de normalizar: {max_val:.6f}")

    # Evitar división por cero o valores muy pequeños
    if max_val > 1e-9:
        final_map = final_map / max_val
    else:
        final_map = xp.zeros_like(final_map)
        if verbose: print("  [DiffusionSim] ADVERTENCIA: Concentración máxima post-simulación es casi cero. Mapa resultante será de ceros.")

    # 7. Preparar Salida 
    if _use_gpu:
        final_map_np = cp.asnumpy(final_map).astype(np.float32)
        obstacle_mask_cpu = cp.asnumpy(obstacle_mask_xp).astype(bool)
    else:
        final_map_np = final_map.astype(np.float32)
        obstacle_mask_cpu = obstacle_mask_xp.astype(bool)


    # 8. Devolver resultados
    return final_map_np, obstacle_mask_cpu