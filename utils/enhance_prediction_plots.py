import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# --- CONFIGURACIÓN ---
PROJECT_ROOT = os.path.expanduser("~/uni/master/tfm/TFM")
THESIS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "thesis_figures_final")
os.makedirs(THESIS_FIGURES_DIR, exist_ok=True)

# Lista de diccionarios con la información de las imágenes
IMAGES_TO_GENERATE = [
    {
        "key": "good_predict_nowind",
        "original_plot_path": "/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparam_search/random_006_h40448/evaluation_test_eval/sample_images/test_ssample_00173_paug_1.png",
        "dataset_folder": "gan_dataset_augmented", 
        "sample_id": "sample_00173", "path_number": "aug_1",
        "title_prefix": "Predicción Exitosa - Escenario Sin Viento"
    },
    {
        "key": "bad_predict_nowind",
        "original_plot_path": "/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparam_search/random_006_h40448/evaluation_test_eval/sample_images/test_ssample_00327_paug_1.png",
        "dataset_folder": "gan_dataset_augmented",
        "sample_id": "sample_00327", "path_number": "aug_1",
        "title_prefix": "Predicción Deficiente - Escenario Sin Viento"
    },
    {
        "key": "good_predict_wind",
        "original_plot_path": "/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparams_search_wind/rs_trial_021_lr0.001_l175.0_gf32/evaluation_test_eval/sample_images/test_ssample_00042_p4.png",
        "dataset_folder": "gan_dataset_wind",
        "sample_id": "sample_00042", "path_number": "4",
        "title_prefix": "Predicción Exitosa - Escenario Con Viento"
    },
    {
        "key": "bad_predict_wind",
        "original_plot_path": "/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparams_search_wind/rs_trial_021_lr0.001_l175.0_gf32/evaluation_test_eval/sample_images/test_ssample_00094_p4.png",
        "dataset_folder": "gan_dataset_wind",
        "sample_id": "sample_00094", "path_number": "4",
        "title_prefix": "Predicción Deficiente - Escenario Con Viento"
    }
]

def main():
    print(f"Generando {len(IMAGES_TO_GENERATE)} figuras para la tesis en: {THESIS_FIGURES_DIR}")
    
    # --- CORRECCIÓN: Iterar sobre una lista, no sobre .items() ---
    for info in tqdm(IMAGES_TO_GENERATE, desc="Generando Figuras"):
        key = info['key'] # Obtener la clave desde dentro del diccionario
        try:
            data_dir = os.path.join(PROJECT_ROOT, "data", info["dataset_folder"])
            metadata_path = os.path.join(data_dir, "metadata.csv")
            original_plot_path = info["original_plot_path"]
            
            if not os.path.exists(metadata_path):
                print(f"\nERROR para '{key}': No se encuentra el archivo de metadatos en {metadata_path}")
                continue
                
            df = pd.read_csv(metadata_path)
            df['path_number_str'] = df['path_number'].astype(str)
            result_df = df[(df['sample_id'] == info['sample_id']) & (df['path_number_str'] == str(info['path_number']))]
            
            if result_df.empty:
                print(f"\nERROR para '{key}': No se encontró entrada para sample_id='{info['sample_id']}' y path_number='{info['path_number']}'.")
                continue
                
            row = result_df.iloc[0]

            obstacle_map = np.load(os.path.join(data_dir, "obstacle_maps", row['obstacle_map_file']))
            gt_map = np.load(os.path.join(data_dir, "ground_truth", row['ground_truth_file']))
            path_df = pd.read_csv(os.path.join(data_dir, "robot_paths", row['robot_path_file']))
            
            if not os.path.exists(original_plot_path):
                print(f"\nADVERTENCIA: No se encontró la imagen de predicción original para {key}: {original_plot_path}")
                continue
            
            original_plot_img = cv2.imread(original_plot_path)
            h, w, _ = original_plot_img.shape
            subplot_width = w // 3
            generated_panel = cv2.cvtColor(original_plot_img[:, subplot_width:2*subplot_width], cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
            
            # Título principal personalizado para cada imagen - mucho más grande
            sample_display = info['sample_id'].replace('sample_', '')
            path_display = f"Ruta {info['path_number']}" if info['path_number'].isdigit() else f"Ruta {info['path_number'].replace('aug_', 'A')}"
            
            main_title = info.get('title_prefix', 'Evaluación del Modelo')
            fig.suptitle(f"{main_title}", 
                        fontsize=24, y=0.95, weight='bold')
            
            # Mostrar el mapa de obstáculos como imagen base
            axes[0].imshow(obstacle_map, cmap='gray', vmin=0, vmax=1)
            
            # Dibujar la ruta del robot usando matplotlib
            coords_x = path_df['pos_j'].values.astype(float)  # coordenadas x (columnas)
            coords_y = path_df['pos_i'].values.astype(float)  # coordenadas y (filas)
            
            if len(coords_x) > 0 and len(coords_y) > 0:
                # Dibujar la ruta completa en verde
                axes[0].plot(coords_x, coords_y, color='lime', linewidth=2, alpha=0.8, label='Trayectoria')
                # Punto inicial en azul
                axes[0].scatter(coords_x[0], coords_y[0], color='blue', s=50, zorder=5, label='Inicio')
                # Punto final en rojo
                axes[0].scatter(coords_x[-1], coords_y[-1], color='red', s=50, zorder=5, label='Final')
                # Añadir leyenda pequeña
                axes[0].legend(loc='upper right', fontsize=12)
            
            axes[0].set_title("Entrada del Modelo\n(Obstáculos y Trayectoria)", fontsize=18, pad=20)
            axes[0].axis('off')
            
            # Panel generado - mejorar visualización
            axes[1].imshow(generated_panel)
            axes[1].set_title("Predicción Generada\npor el Modelo", fontsize=18, pad=20)
            axes[1].axis('off')
            
            # Ground truth con colorbar más consistente
            im = axes[2].imshow(gt_map, cmap='viridis', vmin=0, vmax=1)
            axes[2].set_title("Ground Truth", fontsize=18, pad=20)
            axes[2].axis('off')
            # Colorbar más pequeño y mejor posicionado con etiqueta en español
            cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, shrink=0.8)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Concentración de Gas', rotation=270, labelpad=15, fontsize=14)

            plt.tight_layout(rect=[0, 0.02, 1, 0.92])
            output_filepath = os.path.join(THESIS_FIGURES_DIR, f"{key}.png")
            plt.savefig(output_filepath, dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f"\nERROR al procesar '{key}': {e}")
            
    print("\n¡Figuras finales generadas!")

if __name__ == '__main__':
    main()