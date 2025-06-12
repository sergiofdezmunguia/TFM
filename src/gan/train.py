# src/gan/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import json
import os
import time
from tqdm import tqdm
from torchvision.utils import save_image

from dataset import GasDiffusionDataset 
from models import UNetGenerator, PatchDiscriminator, weights_init_normal

# --- 1. CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.path.expanduser("~/uni/master/tfm/TFM") 
METADATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
PREPROCESSED_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed_for_model")
OUTPUT_RUN_NAME = "gan_training_run_1"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models_outputs", OUTPUT_RUN_NAME)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Modelo
IMG_RESOLUTION = 256 
CONDITION_CHANNELS = 3
TARGET_CHANNELS = 1
GEN_FEATURES = 32    
DISC_FEATURES = 64

# Entrenamiento
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 200     
BATCH_SIZE = 4       
LAMBDA_L1 = 100.0    
AUG_H_FLIP_PROB_TRAIN = 0.5
AUG_V_FLIP_PROB_TRAIN = 0.0 
EARLY_STOPPING_PATIENCE = 20
SAVE_SAMPLE_EVERY_N_EPOCHS = 5
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 25

print(f"--- Configuración de Entrenamiento para '{OUTPUT_RUN_NAME}' ---")
print(f"Dispositivo: {DEVICE}, Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}")
print(f"LR G: {LEARNING_RATE_G}, LR D: {LEARNING_RATE_D}, Lambda L1: {LAMBDA_L1}")
print(f"Gen Features: {GEN_FEATURES}, Disc Features: {DISC_FEATURES}")
print(f"Directorio de Salida: {OUTPUT_DIR}\n")

# --- 2. DATALOADERS ---
try:
    cleaned_df = pd.read_csv(os.path.join(METADATA_BASE_DIR, "cleaned_metadata.csv"))
    with open(os.path.join(METADATA_BASE_DIR, "train_sample_ids.json"), 'r') as f: train_ids = set(json.load(f))
    with open(os.path.join(METADATA_BASE_DIR, "val_sample_ids.json"), 'r') as f: val_ids = set(json.load(f))
except Exception as e: print(f"Error cargando metadatos: {e}"); exit()

train_df = cleaned_df[cleaned_df['sample_id'].isin(train_ids)]
val_df = cleaned_df[cleaned_df['sample_id'].isin(val_ids)]

train_dataset = GasDiffusionDataset(train_df, os.path.join(PREPROCESSED_BASE_DIR, "train", "inputs"), os.path.join(PREPROCESSED_BASE_DIR, "train", "targets"), True, AUG_H_FLIP_PROB_TRAIN, AUG_V_FLIP_PROB_TRAIN)
val_dataset = GasDiffusionDataset(val_df, os.path.join(PREPROCESSED_BASE_DIR, "val", "inputs"), os.path.join(PREPROCESSED_BASE_DIR, "val", "targets"), False)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# --- 3. MODELOS, PÉRDIDAS, OPTIMIZADORES ---
generator = UNetGenerator(CONDITION_CHANNELS, TARGET_CHANNELS, GEN_FEATURES).to(DEVICE)
discriminator = PatchDiscriminator(CONDITION_CHANNELS, TARGET_CHANNELS, DISC_FEATURES).to(DEVICE)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

criterion_GAN = nn.BCEWithLogitsLoss().to(DEVICE)
criterion_L1 = nn.L1Loss().to(DEVICE)

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

writer = SummaryWriter(LOG_DIR)
best_val_loss_L1 = float('inf')
epochs_no_improve = 0

# --- 4. BUCLE DE ENTRENAMIENTO ---
print("\n--- Iniciando Entrenamiento ---")
training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    generator.train()
    discriminator.train()
    
    running_loss_D, running_loss_G, running_loss_G_L1, running_loss_G_GAN = 0.0, 0.0, 0.0, 0.0

    for real_X, real_Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False):
        real_X, real_Y = real_X.to(DEVICE), real_Y.to(DEVICE)

        # Entrenar Discriminador
        optimizer_D.zero_grad()
        fake_Y = generator(real_X)
        disc_real_pred = discriminator(real_X, real_Y)
        loss_D_real = criterion_GAN(disc_real_pred, torch.ones_like(disc_real_pred, device=DEVICE))
        disc_fake_pred = discriminator(real_X, fake_Y.detach())
        loss_D_fake = criterion_GAN(disc_fake_pred, torch.zeros_like(disc_fake_pred, device=DEVICE))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

        # Entrenar Generador
        optimizer_G.zero_grad()
        disc_fake_pred_for_G = discriminator(real_X, fake_Y) # Reusar fake_Y
        loss_G_GAN_adv = criterion_GAN(disc_fake_pred_for_G, torch.ones_like(disc_fake_pred_for_G, device=DEVICE))
        loss_G_L1_recon = criterion_L1(fake_Y, real_Y) * LAMBDA_L1
        loss_G = loss_G_GAN_adv + loss_G_L1_recon
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()
        running_loss_G_L1 += loss_G_L1_recon.item()
        running_loss_G_GAN += loss_G_GAN_adv.item()

    # Log pérdidas de entrenamiento promedio
    avg_loss_D = running_loss_D / len(train_loader)
    avg_loss_G = running_loss_G / len(train_loader)
    avg_loss_G_L1_train = running_loss_G_L1 / len(train_loader)
    avg_loss_G_GAN_train = running_loss_G_GAN / len(train_loader)
    writer.add_scalar("Loss/D_train", avg_loss_D, epoch)
    writer.add_scalar("Loss/G_train_total", avg_loss_G, epoch)
    writer.add_scalar("Loss/G_train_L1", avg_loss_G_L1_train / LAMBDA_L1, epoch) # L1 sin lambda
    writer.add_scalar("Loss/G_train_GAN", avg_loss_G_GAN_train, epoch)

    # Validación
    generator.eval()
    running_val_loss_L1 = 0.0
    with torch.no_grad():
        for batch_idx_val, (val_X, val_Y) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)):
            val_X, val_Y = val_X.to(DEVICE), val_Y.to(DEVICE)
            val_fake_Y = generator(val_X)
            val_loss_L1_batch = criterion_L1(val_fake_Y, val_Y) # Solo L1 para validación de G
            running_val_loss_L1 += val_loss_L1_batch.item()

            if batch_idx_val == 0 and (epoch + 1) % SAVE_SAMPLE_EVERY_N_EPOCHS == 0:
                # Guardar unas pocas imágenes (primeras 4 del batch)
                # Concatenamos el primer canal de real_X (obstáculos), fake_Y, y real_Y
                # Asegúrate que el primer canal de real_X sea interpretable visualmente (ej. obstáculos)
                obstacle_channel = val_X[:4, 0:1, :, :] # Toma el primer canal
                img_sample = torch.cat((obstacle_channel, val_fake_Y[:4], val_Y[:4]), dim=-1) # Concatena horizontalmente
                save_image(img_sample, os.path.join(SAMPLE_DIR, f"val_epoch_{epoch+1}.png"), nrow=1, normalize=True, value_range=(0,1))

    avg_val_loss_L1 = running_val_loss_L1 / len(val_loader)
    writer.add_scalar("Loss/G_val_L1", avg_val_loss_L1, epoch) # L1 real, sin lambda
    
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} [{epoch_duration:.2f}s] - Train D: {avg_loss_D:.4f}, Train G: {avg_loss_G:.4f} (L1: {avg_loss_G_L1_train/LAMBDA_L1:.4f}) | Val L1: {avg_val_loss_L1:.4f}")

    # Checkpoint y Early Stopping
    if avg_val_loss_L1 < best_val_loss_L1:
        best_val_loss_L1 = avg_val_loss_L1
        epochs_no_improve = 0
        torch.save({'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'epoch': epoch, 'val_loss_L1': best_val_loss_L1
                   }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"  Best model saved (Val L1: {best_val_loss_L1:.4f})")
    else:
        epochs_no_improve += 1

    if (epoch + 1) % SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
         torch.save({'generator_state_dict': generator.state_dict(), 'epoch': epoch}, 
                    os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
         print(f"  Periodic checkpoint saved: epoch_{epoch+1}.pth")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1} after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
        break
        
writer.close()
total_training_time = time.time() - training_start_time
print(f"\n--- Entrenamiento Finalizado en {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s ---")
print(f"Mejor L1 de validación del generador: {best_val_loss_L1:.4f}")
print(f"Salidas guardadas en: {OUTPUT_DIR}")