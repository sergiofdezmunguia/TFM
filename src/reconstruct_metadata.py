import pandas as pd
import numpy as np
import os
import re # Para expresiones regulares
from tqdm import tqdm

# --- CONFIGURACIÓN DE DIRECTORIOS ---
# AJUSTA ESTA RUTA a la carpeta principal de tu dataset generado
# (la que contiene las subcarpetas ground_truth, obstacle_maps, robot_paths)
# Ejemplo: /home/sergio/uni/master/tfm/TFM/data/gan_dataset-epsilon_greedy
OUTPUT_PARENT_DIR = os.path.expanduser("~/uni/master/tfm/TFM/data/gan_dataset-epsilon_greedy") 

# Subdirectorios (se construirán a partir de OUTPUT_PARENT_DIR)
OUTPUT_GT_DIR = os.path.join(OUTPUT_PARENT_DIR, "ground_truth")
OUTPUT_OBSTACLES_DIR = os.path.join(OUTPUT_PARENT_DIR, "obstacle_maps")
OUTPUT_PATHS_DIR = os.path.join(OUTPUT_PARENT_DIR, "robot_paths")

# Archivo de salida para los metadatos reconstruidos
RECONSTRUCTED_METADATA_FILE = os.path.join(OUTPUT_PARENT_DIR, "metadata_reconstructed.csv")

# Nombre del mapa PGM base del que se extrajeron todas las ROIs
# Si todas las ROIs vinieron del mapa "demo.pgm", esto es correcto.
BASE_MAP_NAME = "demo"

def reconstruct_metadata():
    print(f"--- Iniciando Reconstrucción Parcial de Metadatos ---")
    print(f"Directorio base del dataset: {OUTPUT_PARENT_DIR}")
    print(f"Buscando Ground Truth en: {OUTPUT_GT_DIR}")
    print(f"Buscando Obstacle Maps en: {OUTPUT_OBSTACLES_DIR}")
    print(f"Buscando Robot Paths en: {OUTPUT_PATHS_DIR}")

    reconstructed_data_list = [] # Lista para almacenar diccionarios, luego se convertirá a DataFrame

    # Verificar que los directorios existen
    if not os.path.isdir(OUTPUT_GT_DIR):
        print(f"ERROR CRÍTICO: Directorio de Ground Truth no encontrado: {OUTPUT_GT_DIR}")
        print("Asegúrate de que la variable OUTPUT_PARENT_DIR esté configurada correctamente.")
        return
    if not os.path.isdir(OUTPUT_OBSTACLES_DIR):
        print(f"ERROR CRÍTICO: Directorio de Obstacle Maps no encontrado: {OUTPUT_OBSTACLES_DIR}")
        return
    if not os.path.isdir(OUTPUT_PATHS_DIR):
        print(f"ERROR CRÍTICO: Directorio de Robot Paths no encontrado: {OUTPUT_PATHS_DIR}")
        return

    # Patrones de expresiones regulares para extraer información de los nombres de archivo
    # Asume nombres como "sample_00001_gt.npy"
    gt_file_pattern = re.compile(r"^(sample_\d+)_gt\.npy$")
    # Asume nombres como "sample_00001_path_0.csv"
    path_file_pattern = re.compile(r"^(sample_\d+)_path_(\d+)\.csv$")

    # 1. Iterar sobre los archivos de Ground Truth para encontrar los sample_ids
    gt_filenames = [f for f in os.listdir(OUTPUT_GT_DIR) if gt_file_pattern.match(f)]
    
    if not gt_filenames:
        print(f"ADVERTENCIA: No se encontraron archivos de Ground Truth que coincidan con el patrón en {OUTPUT_GT_DIR}")
        return
        
    print(f"Encontrados {len(gt_filenames)} archivos de Ground Truth para procesar.")

    for gt_filename_only in tqdm(gt_filenames, desc="Procesando Mapas GT"):
        match_gt = gt_file_pattern.match(gt_filename_only)
        # No es necesario verificar match_gt de nuevo porque ya filtramos la lista
        
        current_sample_id = match_gt.group(1) # ej. "sample_00001"

        # 2. Construir el nombre esperado para el archivo de obstáculos
        obstacle_filename_only = f"{current_sample_id}_obstacles.npy"
        obstacle_filepath_full = os.path.join(OUTPUT_OBSTACLES_DIR, obstacle_filename_only)

        if not os.path.exists(obstacle_filepath_full):
            print(f"ADVERTENCIA: No se encontró el archivo de obstáculos esperado '{obstacle_filename_only}' para {current_sample_id}. Saltando este sample_id.")
            continue
            
        # 3. Buscar todos los archivos de trayectoria (.csv) para este sample_id
        related_path_filenames = []
        for f in os.listdir(OUTPUT_PATHS_DIR):
            if f.startswith(current_sample_id) and f.endswith(".csv") and path_file_pattern.match(f):
                related_path_filenames.append(f)
        
        if not related_path_filenames:
            print(f"ADVERTENCIA: No se encontraron archivos de trayectoria para {current_sample_id}. Saltando este sample_id.")
            continue

        for path_filename_only in related_path_filenames:
            match_path = path_file_pattern.match(path_filename_only)
            # sample_id_from_path = match_path.group(1) # Debería ser igual a current_sample_id
            path_number_str = match_path.group(2)
            
            # 4. Leer el archivo de trayectoria para obtener el número de pasos
            path_filepath_full = os.path.join(OUTPUT_PATHS_DIR, path_filename_only)
            num_path_steps = 0 # Valor por defecto si el archivo no se puede leer
            try:
                path_df = pd.read_csv(path_filepath_full)
                num_path_steps = len(path_df)
            except Exception as e:
                print(f"ERROR al leer o procesar el archivo de path '{path_filepath_full}': {e}. Se usará num_path_steps=0.")
            
            # 5. Añadir la información reconstruida a la lista
            reconstructed_data_list.append({
                'sample_id': current_sample_id,
                'path_number': int(path_number_str),
                'map_name': BASE_MAP_NAME,
                # Columnas que NO PODEMOS RECONSTRUIR con certeza:
                'roi_origin_px_i': np.nan, 
                'roi_origin_px_j': np.nan,
                'source_relative_px_i': np.nan,
                'source_relative_px_j': np.nan,
                # Columnas que SÍ PODEMOS RECONSTRUIR:
                'ground_truth_file': gt_filename_only,         # Solo el nombre del archivo
                'obstacle_map_file': obstacle_filename_only,   # Solo el nombre del archivo
                'robot_path_file': path_filename_only,         # Solo el nombre del archivo
                'num_path_steps': num_path_steps
            })

    # 6. Crear y guardar el DataFrame
    if reconstructed_data_list:
        reconstructed_df = pd.DataFrame(reconstructed_data_list)
        
        # Reordenar columnas para que se parezca al formato original (si es posible/deseado)
        # Esta es una suposición del orden de columnas original. Ajusta si es necesario.
        desired_column_order = [
            'sample_id', 'path_number', 'map_name', 
            'roi_origin_px_i', 'roi_origin_px_j', 
            'source_relative_px_i', 'source_relative_px_j',
            'ground_truth_file', 'obstacle_map_file', 'robot_path_file', 
            'num_path_steps'
        ]
        # Asegurarse de que solo se usan columnas que existen en el DataFrame reconstruido
        actual_column_order = [col for col in desired_column_order if col in reconstructed_df.columns]
        reconstructed_df = reconstructed_df[actual_column_order]

        try:
            reconstructed_df.to_csv(RECONSTRUCTED_METADATA_FILE, index=False)
            print(f"\nMetadatos parcialmente reconstruidos guardados exitosamente en: {RECONSTRUCTED_METADATA_FILE}")
            print(f"Total de entradas (rutas individuales) procesadas y guardadas: {len(reconstructed_df)}")
        except Exception as e:
            print(f"\nERROR CRÍTICO al guardar el archivo CSV de metadatos: {e}")
            print("Verifica los permisos de escritura en el directorio de salida.")
    else:
        print("\nADVERTENCIA: No se pudo reconstruir ninguna entrada de metadatos.")
        print("Verifica que los archivos existen en los directorios especificados y que los patrones de nombres coinciden.")

    print("--- Reconstrucción de Metadatos Finalizada ---")

if __name__ == "__main__":
    # Asegúrate de que la variable OUTPUT_PARENT_DIR esté correctamente configurada arriba
    # antes de ejecutar el script.
    print(f"INFO: Ejecutando script de reconstrucción. OUTPUT_PARENT_DIR está configurado como: '{OUTPUT_PARENT_DIR}'")
    if not os.path.isdir(OUTPUT_PARENT_DIR):
        print(f"ERROR FATAL: La ruta especificada en OUTPUT_PARENT_DIR ('{OUTPUT_PARENT_DIR}') no es un directorio válido o no existe.")
        print("Por favor, corrige la ruta al inicio del script y vuelve a intentarlo.")
    else:
        reconstruct_metadata()