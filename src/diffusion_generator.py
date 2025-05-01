import numpy as np
import time

try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False
    def tqdm(iterable, **kwargs):
        print("INFO: tqdm no instalado, no se mostrará barra de progreso.")
        return iterable

try:
    import cupy as cp
    if not cp.is_available(): raise ImportError("CuPy instalado pero no detecta GPU CUDA.")
    _use_gpu = True
    print("INFO: Usando CuPy (GPU) para la simulación.")
except ImportError:
    cp = np 
    _use_gpu = False
    print("INFO: Usando NumPy (CPU) para la simulación.")

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

def generate_diffusion_map_roi(
    map_subsection_np, 
    source_coords_px_relative,
    occupied_thresh_prob = DEFAULT_OCCUPIED_THRESH_PROB,
    timesteps=DEFAULT_TIMESTEPS,
    diffusion_rate=DEFAULT_DIFFUSION_RATE,
    dissipation_rate=DEFAULT_DISSIPATION_RATE,
    source_strength=DEFAULT_SOURCE_STRENGTH,
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
        verbose (bool): Si imprimir mensajes de progreso/debug internos.

    Returns:
        tuple: (final_map_np, obstacle_mask)
               - final_map_np (np.ndarray | None): Mapa de concentración final normalizado (0-1) en CPU (float32), o None si hay error.
               - obstacle_mask (np.ndarray | None): Máscara booleana de obstáculos (True=Obstáculo) en CPU, o None si hay error.
               Devuelve (None, None) si la fuente está en un obstáculo o error inicial.
    """
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
        return None, None

    # 2. Calcular Umbral y Crear Máscara de Obstáculos
    try:
        OBSTACLE_PGM_MAX_VALUE = get_occupied_pgm_threshold(occupied_thresh_prob)
        obstacle_mask_xp = xp.asarray(map_subsection_np <= OBSTACLE_PGM_MAX_VALUE)
    except Exception as e:
        print(f"ERROR [DiffusionSim]: Creando máscara de obstáculos: {e}", file=sys.stderr)
        return None, None

    # 3. Validar Píxel Fuente 
    try:
        is_source_obstacle = obstacle_mask_xp[source_i, source_j]
        if isinstance(is_source_obstacle, cp.ndarray):
            is_source_obstacle = bool(is_source_obstacle.get())
        else:
            is_source_obstacle = bool(is_source_obstacle)

        if is_source_obstacle:
            if verbose: print(f"ADVERTENCIA [DiffusionSim]: Píxel fuente ({source_i},{source_j}) está en un obstáculo (Valor PGM {map_subsection_np[source_i, source_j]} <= {OBSTACLE_PGM_MAX_VALUE}). No se ejecutará simulación.")
            obstacle_mask_cpu = cp.asnumpy(obstacle_mask_xp) if _use_gpu else obstacle_mask_xp
            return None, obstacle_mask_cpu.astype(bool)
    except Exception as e:
        print(f"ERROR [DiffusionSim]: Validando píxel fuente: {e}", file=sys.stderr)
        return None, None


    # 4. Inicializar Grid de Concentración
    concentration_grid = xp.zeros((height, width), dtype=xp.float32)
    # Añadir fuente inicial solo si no es obstáculo
    concentration_grid[source_i, source_j] = xp.float32(source_strength)
    if verbose: print(f"  [DiffusionSim] DEBUG: Added initial source strength {source_strength} at ({source_i},{source_j})")

    # 5. Bucle Principal de Simulación (Método explícito FTCS con Laplaciano 5 puntos)
    loop_iterator = range(timesteps)
    if verbose:
        try:
            from tqdm import tqdm as internal_tqdm
            loop_iterator = internal_tqdm(loop_iterator, desc="Simulación Difusión Interna", leave=False, disable=False)
        except ImportError:
            pass

    try:
        for _ in loop_iterator:
            current_grid = xp.copy(concentration_grid)

            # Calcular Laplaciano en puntos interiores
            laplacian = (current_grid[:-2, 1:-1] + current_grid[2:, 1:-1] +  # Arriba, Abajo
                         current_grid[1:-1, :-2] + current_grid[1:-1, 2:] -  # Izquierda, Derecha
                         4 * current_grid[1:-1, 1:-1])                       # Centro * 4

            # Calcular cambio debido a difusión
            change = diffusion_rate * laplacian

            # Máscara de puntos interiores que NO son obstáculos
            non_obstacle_interior_mask = ~obstacle_mask_xp[1:-1, 1:-1]

            # Actualizar concentración SOLO en puntos interiores no-obstáculo
            concentration_grid[1:-1, 1:-1][non_obstacle_interior_mask] += change[non_obstacle_interior_mask]

            # Aplicar Disipación en toda la grid
            if dissipation_rate > 1e-9: # Evitar multiplicación innecesaria si es 0
                concentration_grid *= (1.0 - dissipation_rate)

            # Forzar concentración a CERO en todos los obstáculos (incluyendo bordes)
            concentration_grid[obstacle_mask_xp] = 0.0

            # Asegurar que la concentración no sea negativa
            xp.maximum(concentration_grid, 0.0, out=concentration_grid)

    except Exception as e:
         print(f"\nERROR [DiffusionSim]: Durante bucle de simulación: {e}", file=sys.stderr)
         obstacle_mask_cpu = cp.asnumpy(obstacle_mask_xp) if _use_gpu else obstacle_mask_xp
         return None, obstacle_mask_cpu.astype(bool)

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