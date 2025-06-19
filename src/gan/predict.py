import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from torchvision.utils import save_image # O usar matplotlib para guardar

# Importar tus módulos
from dataset import GasDiffusionDataset
from models import UNetGenerator

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

PROJECT_ROOT = os.path.expanduser("~/uni/master/tfm/TFM") 
METADATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "metadata", "augmented_cleaned")
PREPROCESSED_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed_for_model_augmented") 

# Directorio de la ejecución de entrenamiento de donde cargar el modelo
TRAINING_RUN_NAME = "gan_training_run_3"
MODEL_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models_outputs", TRAINING_RUN_NAME, "checkpoints")
BEST_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")

# Directorio de salida para la evaluación
EVALUATION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models_outputs", TRAINING_RUN_NAME, "evaluation_on_test")
EVAL_IMAGES_DIR = os.path.join(EVALUATION_OUTPUT_DIR, "sample_images")
EVAL_RESULTS_FILE = os.path.join(EVALUATION_OUTPUT_DIR, "test_metrics_results.txt")

os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_IMAGES_DIR, exist_ok=True)

# Hiperparámetros del modelo
CONDITION_CHANNELS = 3
TARGET_CHANNELS = 1
GEN_FEATURES = 64
IMG_RESOLUTION = 256

# Parámetros de evaluación
BATCH_SIZE_EVAL = 4 
NUM_SAMPLES_TO_VISUALIZE = 20
IOU_THRESHOLD = 0.25

# --- Funciones Auxiliares de Métricas ---
def calculate_mae(generated, target):
    return np.mean(np.abs(generated - target))

def calculate_mse(generated, target):
    return np.mean((generated - target)**2)

def calculate_psnr(generated, target, data_range=1.0):
    generated_clipped = np.clip(generated, 0, 1)
    target_clipped = np.clip(target, 0, 1)
    return peak_signal_noise_ratio(target_clipped, generated_clipped, data_range=data_range)

def calculate_ssim(generated, target, data_range=1.0, win_size=7):
    generated_squeezed = generated.squeeze(axis=-1) if generated.ndim == 3 and generated.shape[-1] == 1 else generated
    target_squeezed = target.squeeze(axis=-1) if target.ndim == 3 and target.shape[-1] == 1 else target

    actual_win_size = min(win_size, generated_squeezed.shape[0], generated_squeezed.shape[1])
    if actual_win_size % 2 == 0:
        actual_win_size -= 1
    if actual_win_size < 3:
        print(f"ADVERTENCIA: Tamaño de ventana para SSIM ({actual_win_size}) es muy pequeño. SSIM podría no ser significativo.")
        return np.nan

    return structural_similarity(target_squeezed, generated_squeezed, data_range=data_range, win_size=actual_win_size, channel_axis=None) # channel_axis=None si es monocanal

def get_peak_coords(image_map):
    if image_map.ndim == 3 and image_map.shape[-1] == 1:
        image_map = image_map.squeeze(axis=-1)
    if image_map.size == 0: return (np.nan, np.nan)
    coords = np.unravel_index(np.argmax(image_map, axis=None), image_map.shape)
    return (coords[0], coords[1])

def calculate_peak_distance(peak_coords_gt, peak_coords_gen):
    if any(np.isnan(c) for c in peak_coords_gt) or any(np.isnan(c) for c in peak_coords_gen):
        return np.nan
    return np.sqrt((peak_coords_gt[0] - peak_coords_gen[0])**2 + (peak_coords_gt[1] - peak_coords_gen[1])**2)

def calculate_peak_intensity_error(peak_val_gt, peak_val_gen):
    return np.abs(peak_val_gt - peak_val_gen)

def calculate_iou(mask_gt, mask_gen):
    intersection = np.logical_and(mask_gt, mask_gen)
    union = np.logical_or(mask_gt, mask_gen)
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)

# --- Función Principal de Evaluación ---
def evaluate_model():
    print(f"--- Iniciando Evaluación del Modelo en el Conjunto de Test ---")
    print(f"Cargando modelo desde: {BEST_MODEL_PATH}")
    print(f"Resultados se guardarán en: {EVALUATION_OUTPUT_DIR}")

    # 1. Cargar Datos de Test
    try:
        cleaned_df = pd.read_csv(os.path.join(METADATA_BASE_DIR, "cleaned_metadata.csv"))
        with open(os.path.join(METADATA_BASE_DIR, "test_sample_ids.json"), 'r') as f: test_ids = set(json.load(f))
    except Exception as e: print(f"Error cargando archivos de metadatos de test: {e}"); exit()

    test_df = cleaned_df[cleaned_df['sample_id'].isin(test_ids)]
    if test_df.empty: print("No hay datos de test para evaluar."); exit()

    test_input_dir_X = os.path.join(PREPROCESSED_BASE_DIR, "test", "inputs")
    test_target_dir_Y = os.path.join(PREPROCESSED_BASE_DIR, "test", "targets")

    test_dataset = GasDiffusionDataset(test_df, test_input_dir_X, test_target_dir_Y, apply_augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset de Test: {len(test_dataset)} muestras, DataLoader: {len(test_loader)} batches")

    # 2. Cargar Modelo Generador Entrenado
    generator = UNetGenerator(CONDITION_CHANNELS, TARGET_CHANNELS, GEN_FEATURES).to(DEVICE)
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"ERROR: No se encontró el archivo del modelo entrenado en {BEST_MODEL_PATH}"); exit()
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval() # ¡Muy importante poner en modo evaluación!
    print(f"Modelo Generador cargado. Entrenado por {checkpoint.get('epoch', 'N/A')+1} épocas. Val L1: {checkpoint.get('val_loss_L1', 'N/A'):.4f}")

    # 3. Listas para almacenar métricas de todas las muestras
    all_mae, all_mse, all_psnr, all_ssim = [], [], [], []
    all_peak_dist, all_peak_intensity_err, all_iou = [], [], []
    
    # 4. Bucle de Evaluación
    print("\nEvaluando en el conjunto de test...")
    samples_visualized_count = 0
    with torch.no_grad():
        for batch_idx, (real_X_batch, real_Y_batch) in enumerate(tqdm(test_loader, desc="Evaluando Test")):
            real_X_batch = real_X_batch.to(DEVICE)

            fake_Y_batch = generator(real_X_batch) # (B, 1, H, W)

            real_X_np_batch = real_X_batch.cpu().numpy().transpose(0, 2, 3, 1) # (B, H, W, 3)
            fake_Y_np_batch = fake_Y_batch.cpu().numpy().transpose(0, 2, 3, 1)   # (B, H, W, 1)
            real_Y_np_batch = real_Y_batch.cpu().numpy().transpose(0, 2, 3, 1)   # (B, H, W, 1)

            for i in range(real_X_np_batch.shape[0]):
                real_X_sample = real_X_np_batch[i]     # (H, W, 3)
                fake_Y_sample = fake_Y_np_batch[i]     # (H, W, 1)
                real_Y_sample = real_Y_np_batch[i]     # (H, W, 1)

                # Calcular métricas
                all_mae.append(calculate_mae(fake_Y_sample, real_Y_sample))
                all_mse.append(calculate_mse(fake_Y_sample, real_Y_sample))
                all_psnr.append(calculate_psnr(fake_Y_sample, real_Y_sample))
                ssim_val = calculate_ssim(fake_Y_sample, real_Y_sample)
                if not np.isnan(ssim_val): all_ssim.append(ssim_val)


                peak_coords_gt = get_peak_coords(real_Y_sample)
                peak_coords_gen = get_peak_coords(fake_Y_sample)
                all_peak_dist.append(calculate_peak_distance(peak_coords_gt, peak_coords_gen))
                
                if not np.isnan(peak_coords_gt[0]) and not np.isnan(peak_coords_gen[0]):
                    peak_val_gt = real_Y_sample[peak_coords_gt[0], peak_coords_gt[1], 0]
                    peak_val_gen = fake_Y_sample[peak_coords_gen[0], peak_coords_gen[1], 0]
                    all_peak_intensity_err.append(calculate_peak_intensity_error(peak_val_gt, peak_val_gen))

                mask_gt = real_Y_sample.squeeze() > IOU_THRESHOLD
                mask_gen = fake_Y_sample.squeeze() > IOU_THRESHOLD
                all_iou.append(calculate_iou(mask_gt, mask_gen))

                # Guardar visualizaciones de algunas muestras
                if samples_visualized_count < NUM_SAMPLES_TO_VISUALIZE:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(real_X_sample[:,:,0], cmap='gray', vmin=0, vmax=1) 
                    axes[0].set_title(f"Entrada (Obstáculos)\nSample: {test_df.iloc[batch_idx * BATCH_SIZE_EVAL + i]['sample_id']}_p{test_df.iloc[batch_idx * BATCH_SIZE_EVAL + i]['path_number']}")
                    axes[0].axis('off')

                    im_gen = axes[1].imshow(fake_Y_sample.squeeze(), cmap='viridis', vmin=0, vmax=1)
                    axes[1].set_title("Heatmap Generado")
                    axes[1].axis('off')
                    
                    im_real = axes[2].imshow(real_Y_sample.squeeze(), cmap='viridis', vmin=0, vmax=1)
                    axes[2].set_title("Heatmap Ground Truth")
                    axes[2].axis('off')
                    
                    fig.colorbar(im_gen, ax=axes[1], fraction=0.046, pad=0.04)
                    fig.colorbar(im_real, ax=axes[2], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    sample_filename = f"test_eval_sample_{batch_idx * BATCH_SIZE_EVAL + i}.png"
                    plt.savefig(os.path.join(EVAL_IMAGES_DIR, sample_filename))
                    plt.close(fig)
                    samples_visualized_count += 1
    
    # 5. Calcular y Mostrar/Guardar Métricas Agregadas
    valid_peak_dist = [d for d in all_peak_dist if not np.isnan(d)]
    
    results_summary = f"--- Resultados de Evaluación en Conjunto de Test ({len(test_dataset)} muestras) ---\n"
    results_summary += f"MAE (L1):                {np.mean(all_mae):.4f} +/- {np.std(all_mae):.4f}\n"
    results_summary += f"MSE:                     {np.mean(all_mse):.4f} +/- {np.std(all_mse):.4f}\n"
    results_summary += f"PSNR:                    {np.mean(all_psnr):.2f} dB +/- {np.std(all_psnr):.2f} dB\n"
    if all_ssim:
        results_summary += f"SSIM:                    {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}\n"
    else:
        results_summary += f"SSIM:                    N/A (posiblemente por tamaño de ventana)\n"

    results_summary += "\nMétricas Específicas del Dominio:\n"
    if valid_peak_dist:
        results_summary += f"Distancia al Pico (px):  {np.mean(valid_peak_dist):.2f} px +/- {np.std(valid_peak_dist):.2f} px\n"
    else:
        results_summary += f"Distancia al Pico (px):  N/A (no se pudieron calcular distancias válidas)\n"
    if all_peak_intensity_err:
        results_summary += f"Error Intensidad Pico:   {np.mean(all_peak_intensity_err):.4f} +/- {np.std(all_peak_intensity_err):.4f}\n"
    else:
        results_summary += f"Error Intensidad Pico:   N/A\n"
    results_summary += f"IoU (umbral {IOU_THRESHOLD:.2f}):      {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}\n"
    
    print("\n" + results_summary)
    with open(EVAL_RESULTS_FILE, 'w') as f:
        f.write(results_summary)
    print(f"Resultados guardados en: {EVAL_RESULTS_FILE}")
    print(f"Imágenes de muestra guardadas en: {EVAL_IMAGES_DIR}")
    print("--- Evaluación Finalizada ---")

if __name__ == '__main__':
    evaluate_model()