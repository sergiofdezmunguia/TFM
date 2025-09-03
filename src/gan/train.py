import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter
import pandas as pd, json, os, time, matplotlib.pyplot as plt, cv2, numpy as np
from tqdm import tqdm
import argparse

from dataset import GasDiffusionDataset 
from models import UNetGenerator, PatchDiscriminator, weights_init_normal

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return default_collate(batch)

def get_args():
    parser = argparse.ArgumentParser(description="Entrenamiento GAN para difusión de gas.")
    parser.add_argument("--project_root", type=str, default=os.path.expanduser("~/uni/master/tfm/TFM"))
    parser.add_argument("--metadata_subdir", type=str, default="data/metadata/wind_cleaned")
    parser.add_argument("--preprocessed_subdir", type=str, default="data/processed_for_model_wind")
    parser.add_argument("--path_csv_dataset_name", type=str, default="data/gan_dataset_wind")
    parser.add_argument("--output_base_dir", type=str, default=None)
    parser.add_argument("--output_run_name", type=str, default="gan_training_default")
    parser.add_argument("--condition_channels", type=int, default=5)
    parser.add_argument("--target_channels", type=int, default=1)
    parser.add_argument("--gen_features", type=int, default=64)
    parser.add_argument("--disc_features", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lambda_l1", type=float, default=75.0)
    parser.add_argument("--h_flip_prob", type=float, default=0.5)
    parser.add_argument("--v_flip_prob", type=float, default=0.5)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--save_sample_every", type=int, default=5)
    parser.add_argument("--save_checkpoint_every", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()

def train_gan(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    METADATA_BASE_DIR = os.path.join(args.project_root, args.metadata_subdir)
    PREPROCESSED_BASE_DIR = os.path.join(args.project_root, args.preprocessed_subdir)
    ORIGINAL_PATHS_CSV_DIR = os.path.join(args.project_root, args.path_csv_dataset_name, "robot_paths")

    output_base = args.output_base_dir if args.output_base_dir is not None else os.path.join(args.project_root, "models_outputs")
    OUTPUT_DIR = os.path.join(output_base, args.output_run_name)
    CHECKPOINT_DIR, SAMPLE_DIR, LOG_DIR = [os.path.join(OUTPUT_DIR, d) for d in ["checkpoints","samples","logs"]]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(SAMPLE_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"--- Configuración para '{args.output_run_name}' ---")
    
    cleaned_df=pd.read_csv(os.path.join(METADATA_BASE_DIR,"cleaned_metadata.csv"))
    with open(os.path.join(METADATA_BASE_DIR,"train_sample_ids.json"),'r') as f: train_ids=set(json.load(f))
    with open(os.path.join(METADATA_BASE_DIR,"val_sample_ids.json"),'r') as f: val_ids=set(json.load(f))
    train_df=cleaned_df[cleaned_df['sample_id'].isin(train_ids)]; val_df=cleaned_df[cleaned_df['sample_id'].isin(val_ids)]
    
    train_dataset = GasDiffusionDataset(train_df, os.path.join(PREPROCESSED_BASE_DIR,"train","inputs"), os.path.join(PREPROCESSED_BASE_DIR,"train","targets"), ORIGINAL_PATHS_CSV_DIR, True, args.h_flip_prob, args.v_flip_prob)
    val_dataset = GasDiffusionDataset(val_df, os.path.join(PREPROCESSED_BASE_DIR,"val","inputs"), os.path.join(PREPROCESSED_BASE_DIR,"val","targets"), ORIGINAL_PATHS_CSV_DIR, False)
    
    train_loader=DataLoader(train_dataset,args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate_fn_skip_none,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_dataset,args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate_fn_skip_none,pin_memory=True)
    print(f"Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    generator=UNetGenerator(args.condition_channels,args.target_channels,args.gen_features).to(DEVICE)
    discriminator=PatchDiscriminator(args.condition_channels,args.target_channels,args.disc_features).to(DEVICE)
    generator.apply(weights_init_normal); discriminator.apply(weights_init_normal)
    
    criterion_GAN=nn.BCEWithLogitsLoss().to(DEVICE); criterion_L1=nn.L1Loss().to(DEVICE)
    optimizer_G=optim.Adam(generator.parameters(),lr=args.lr_g,betas=(args.beta1,args.beta2)); optimizer_D=optim.Adam(discriminator.parameters(),lr=args.lr_d,betas=(args.beta1,args.beta2))
    
    writer = SummaryWriter(LOG_DIR); best_val_loss_L1 = float('inf'); epochs_no_improve = 0
    print("\n--- Iniciando Entrenamiento ---")
    
    for epoch in range(args.num_epochs):
        generator.train(); discriminator.train()
        epoch_loss_D, epoch_loss_G, epoch_loss_L1, epoch_loss_GAN = 0.0, 0.0, 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"E {epoch+1}/{args.num_epochs} [Train]", leave=False):
            if batch is None: continue
            real_X, real_Y, _, _, _ = batch
            real_X, real_Y = real_X.to(DEVICE), real_Y.to(DEVICE)
            
            optimizer_D.zero_grad();
            with torch.no_grad(): fake_Y = generator(real_X)
            loss_D=(criterion_GAN(discriminator(real_X,real_Y),torch.ones_like(discriminator(real_X,real_Y)))+criterion_GAN(discriminator(real_X,fake_Y.detach()),torch.zeros_like(discriminator(real_X,fake_Y.detach()))))/2
            loss_D.backward(); optimizer_D.step(); epoch_loss_D += loss_D.item()
            
            optimizer_G.zero_grad(); fake_Y_for_G=generator(real_X)
            loss_G_adv=criterion_GAN(discriminator(real_X,fake_Y_for_G),torch.ones_like(discriminator(real_X,fake_Y_for_G)))
            loss_G_L1=criterion_L1(fake_Y_for_G,real_Y)*args.lambda_l1
            loss_G=loss_G_adv+loss_G_L1; loss_G.backward(); optimizer_G.step()
            epoch_loss_G += loss_G.item(); epoch_loss_L1 += loss_G_L1.item(); epoch_loss_GAN += loss_G_adv.item()

        generator.eval(); epoch_val_L1=0.0
        with torch.no_grad():
            for batch_idx, batch_val in enumerate(val_loader):
                if batch_val is None: continue
                val_X_b, val_Y_b, val_paths_coords_b, val_s_ids_b, val_p_ids_b = batch_val
                val_X,val_Y=val_X_b.to(DEVICE),val_Y_b.to(DEVICE)
                val_fake_Y=generator(val_X); epoch_val_L1+=criterion_L1(val_fake_Y,val_Y).item()

                if batch_idx == 0 and (epoch + 1) % args.save_sample_every == 0:
                    current_batch_size = val_X.size(0)
                    for i in range(min(4, current_batch_size)):
                        global_idx = batch_idx * args.batch_size + i
                        if global_idx >= len(val_df): continue
                        
                        row = val_df.iloc[global_idx]
                        s_id, p_id = str(row['sample_id']), str(row['path_number'])
                        csv_filename = str(row['robot_path_file'])
                        
                        # --- Carga de coordenadas (esta parte ya era correcta) ---
                        path_coords_from_csv = []
                        csv_path = os.path.join(ORIGINAL_PATHS_CSV_DIR, csv_filename)
                        if os.path.exists(csv_path):
                            try:
                                path_df = pd.read_csv(csv_path)
                                if 'pos_j' in path_df.columns and 'pos_i' in path_df.columns:
                                    coords = zip(path_df['pos_j'].values, path_df['pos_i'].values)
                                    path_coords_from_csv = [(int(x), int(y)) for x, y in coords if not np.isnan(x)]
                            except:
                                pass
                        
                        # --- LÓGICA DE VISUALIZACIÓN ---
                        # Extraer los 3 (o 5) canales de entrada a la CPU
                        input_channels_np = val_X[i].cpu().numpy() # Forma: (C, H, W)
                        
                        # Transponer a formato de imagen (H, W, C)
                        input_display_base = input_channels_np.transpose(1, 2, 0)
                        
                        # Crear una imagen RGB a partir de los canales para una mejor visualización
                        # Asumimos: Canal 0=Obstáculos, 1=Máscara Ruta, 2=Detecciones Gas
                        # Si tienes 5 canales, puedes decidir cuáles mostrar
                        obstacle_ch = input_display_base[:, :, 0]
                        path_mask_ch = input_display_base[:, :, 1]
                        detection_ch = input_display_base[:, :, 2]
                        
                        # Combinar en una imagen a color: R=Detecciones, G=Ruta, B=Obstáculos
                        input_display_rgb = np.stack([
                            detection_ch,       # Canal Rojo
                            path_mask_ch,       # Canal Verde
                            obstacle_ch         # Canal Azul
                        ], axis=-1).astype(np.float32)

                        # Dibujar la ruta cargada del CSV encima con un color de alto contraste para confirmación
                        if path_coords_from_csv and len(path_coords_from_csv) > 1:
                            pts = np.array(path_coords_from_csv, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(input_display_rgb, [pts], isClosed=False, color=(0, 1, 1), thickness=2) # Ruta en Cian
                        
                        # Extraer heatmaps
                        fake_np = val_fake_Y[i, 0].cpu().numpy()
                        real_np = val_Y[i, 0].cpu().numpy()
                        
                        # Crear la figura
                        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Un poco más ancho
                        fig.suptitle(f"Epoch {epoch+1} - Val Sample: {s_id}_p{p_id}", fontsize=14)
                        
                        # Mostrar la nueva imagen de entrada combinada
                        axes[0].imshow(np.clip(input_display_rgb, 0, 1))
                        axes[0].set_title("Input (G:Path, B:Obst)")
                        axes[0].axis('off')
                        
                        # Mostrar heatmaps
                        im1 = axes[1].imshow(fake_np, cmap='viridis', vmin=0, vmax=1)
                        axes[1].set_title("Generated"); axes[1].axis('off')
                        fig.colorbar(im1, ax=axes[1])
                        
                        im2 = axes[2].imshow(real_np, cmap='viridis', vmin=0, vmax=1)
                        axes[2].set_title("Ground Truth"); axes[2].axis('off')
                        fig.colorbar(im2, ax=axes[2])
                        
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.savefig(os.path.join(SAMPLE_DIR, f"val_e{epoch+1}_s{s_id}_p{p_id}.png"))
                        plt.close(fig)
        
        avg_val_L1=epoch_val_L1/len(val_loader) if len(val_loader)>0 else float('inf')
        writer.add_scalar("Loss/G_val_L1",avg_val_L1,epoch);print(f"E {epoch+1} - Train D:{epoch_loss_D/len(train_loader):.4f} | Val L1:{avg_val_L1:.4f}")
        if avg_val_L1<best_val_loss_L1:best_val_loss_L1=avg_val_L1;epochs_no_improve=0;torch.save({'generator_state_dict':generator.state_dict(),'epoch':epoch,'val_loss_L1':best_val_loss_L1,'args':vars(args)},os.path.join(CHECKPOINT_DIR,"best_model.pth"));print(f"  Best saved (Val L1:{best_val_loss_L1:.4f})")
        else:epochs_no_improve+=1
        if epochs_no_improve>=args.early_stopping_patience:print(f"Early stopping @ E {epoch+1}");break
    writer.close()

if __name__ == "__main__":
    cli_args = get_args()
    train_gan(cli_args)