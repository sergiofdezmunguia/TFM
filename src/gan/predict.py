import torch
from torch.utils.data import DataLoader, default_collate
import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import cv2
import argparse

# Asumiendo que dataset.py y models.py están accesibles
from dataset import GasDiffusionDataset
from models import UNetGenerator

# --- FUNCIÓN COLLATE (CRUCIAL PARA MANEJAR ERRORES DE CARGA) ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return default_collate(batch)

# --- Funciones de Métricas ---
def calculate_mae(gen,tgt): 
    return np.mean(np.abs(gen-tgt))
def calculate_mse(gen,tgt): 
    return np.mean((gen-tgt)**2)
def calculate_psnr(gen,tgt,dr=1.0): 
    gen_c,tgt_c=np.clip(gen,0,1),np.clip(tgt,0,1)
    return peak_signal_noise_ratio(tgt_c,gen_c,data_range=dr)
def calculate_ssim(gen,tgt,dr=1.0,ws=7):
    g_sq,t_sq=(x.squeeze()if x.ndim==3 and x.shape[-1]==1 else x for x in(gen,tgt))
    aw=min(ws,g_sq.shape[0],g_sq.shape[1])
    aw=aw-1 if aw%2==0 else aw
    if aw < 3: 
        return np.nan
    return structural_similarity(t_sq,g_sq,data_range=dr,win_size=aw,channel_axis=None)
def get_peak_coords(m):
    m_sq=m.squeeze()if m.ndim==3 and m.shape[-1]==1 else m
    if m_sq.size==0: 
        return (np.nan, np.nan)
    return np.unravel_index(np.argmax(m_sq),m_sq.shape)
def calculate_peak_distance(pg,pp):
    if any(np.isnan(c) for c_list in(pg,pp) for c in c_list): 
        return np.nan
    return np.sqrt(sum((gc-pc)**2 for gc,pc in zip(pg,pp)))
def calculate_peak_intensity_error(vg,vp): return np.abs(vg-vp)
def calculate_iou(mg,mp):
    i=np.logical_and(mg,mp); u=np.logical_or(mg,mp)
    if np.sum(u)==0: return 1.0 if np.sum(i)==0 else 0.0
    return np.sum(i)/np.sum(u)

def get_args_predict():
    parser = argparse.ArgumentParser(description="Evaluación del modelo GAN.")
    parser.add_argument("--project_root",type=str,default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
    parser.add_argument("--models_base_dir",type=str,default=None, help="Directorio base donde se encuentran las corridas de entrenamiento.")
    parser.add_argument("--training_run_name",type=str,required=True, help="Nombre de la corrida de entrenamiento a evaluar.")
    parser.add_argument("--eval_run_suffix",type=str,default="eval")
    parser.add_argument("--metadata_subdir",type=str,default="data/metadata/wind_cleaned")
    parser.add_argument("--preprocessed_subdir",type=str,default="data/processed_for_model_wind")
    parser.add_argument("--path_csv_dataset_name",type=str,default="data/gan_dataset_wind")
    parser.add_argument("--condition_channels", type=int, default=3)
    parser.add_argument("--target_channels", type=int, default=1)
    parser.add_argument("--gen_features",type=int,default=64,help="Debe coincidir con el modelo cargado.")
    parser.add_argument("--batch_size_eval",type=int,default=4)
    parser.add_argument("--iou_threshold",type=float,default=0.25)
    parser.add_argument("--num_samples_to_visualize",type=int,default=20)
    parser.add_argument("--num_workers",type=int,default=2)
    return parser.parse_args()

def evaluate_model_main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_base = args.models_base_dir if args.models_base_dir is not None else os.path.join(args.project_root, "models_outputs")
    MODEL_CHECKPOINT_DIR = os.path.join(models_base, args.training_run_name, "checkpoints")
    BEST_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
    EVALUATION_OUTPUT_DIR = os.path.join(models_base, args.training_run_name, f"evaluation_test_{args.eval_run_suffix}")
    EVAL_IMAGES_DIR = os.path.join(EVALUATION_OUTPUT_DIR, "sample_images")
    METRICS_JSON_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "test_metrics_summary.json")

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True); os.makedirs(EVAL_IMAGES_DIR, exist_ok=True)
    print(f"--- Evaluando Modelo de '{args.training_run_name}' ---")
    
    METADATA_DIR = os.path.join(args.project_root,args.metadata_subdir)
    PREPROCESSED_DIR = os.path.join(args.project_root,args.preprocessed_subdir)
    PATHS_CSV_DIR = os.path.join(args.project_root,args.path_csv_dataset_name,"robot_paths")
    
    df=pd.read_csv(os.path.join(METADATA_DIR,"cleaned_metadata.csv"));
    with open(os.path.join(METADATA_DIR,"test_sample_ids.json"),'r') as f:test_ids=set(json.load(f))
    test_df = df[df['sample_id'].isin(test_ids)].reset_index(drop=True)
    
    test_dataset=GasDiffusionDataset(test_df, os.path.join(PREPROCESSED_DIR,"test","inputs"), os.path.join(PREPROCESSED_DIR,"test","targets"), PATHS_CSV_DIR, apply_augmentation=False)
    test_loader=DataLoader(test_dataset, args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    print(f"Test samples: {len(test_df)}")
    
    if not os.path.exists(BEST_MODEL_PATH): print(f"ERROR: Modelo no encontrado {BEST_MODEL_PATH}"); return
    
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    train_args = ckpt.get('args', {})
    gen_feat_ckpt = train_args.get('gen_features', args.gen_features)
    cond_channels_ckpt = train_args.get('condition_channels', args.condition_channels)
    generator=UNetGenerator(cond_channels_ckpt, args.target_channels, gen_feat_ckpt).to(DEVICE)
    generator.load_state_dict(ckpt['generator_state_dict']); generator.eval()
    print(f"Modelo cargado. Epoch: {ckpt.get('epoch','N/A')+1}, ValL1: {ckpt.get('val_loss_L1','N/A'):.4f}, GenFeat: {gen_feat_ckpt}")

    all_metrics = {k:[] for k in ["mae","mse","psnr","ssim","peak_dist","peak_int_err","iou"]}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluando Test")):
            if batch is None: continue
            
            real_X_b, real_Y_b, paths_coords_b, s_ids_b, p_ids_b = batch
            fake_Y_tensor = generator(real_X_b.to(DEVICE))
            
            current_batch_size = real_X_b.size(0)
            for i in range(current_batch_size):
                global_idx = batch_idx * args.batch_size_eval + i
                if global_idx >= len(test_df): continue
                
                fy_np, ry_np = fake_Y_tensor[i,0].cpu().numpy(), real_Y_b[i,0].cpu().numpy()
                
                # --- CÁLCULO DE MÉTRICAS ---
                all_metrics["mae"].append(calculate_mae(fy_np, ry_np))
                all_metrics["mse"].append(calculate_mse(fy_np, ry_np))
                all_metrics["psnr"].append(calculate_psnr(fy_np, ry_np))
                ssim_val = calculate_ssim(fy_np, ry_np)
                if not np.isnan(ssim_val): all_metrics["ssim"].append(ssim_val)
                pgt,pgen=get_peak_coords(ry_np),get_peak_coords(fy_np)
                all_metrics["peak_dist"].append(calculate_peak_distance(pgt,pgen))
                if not np.any(np.isnan(pgt)) and not np.any(np.isnan(pgen)):
                    all_metrics["peak_int_err"].append(calculate_peak_intensity_error(ry_np[pgt],fy_np[pgen]))
                mask_gt,mask_gen=(ry_np > args.iou_threshold),(fy_np > args.iou_threshold)
                all_metrics["iou"].append(calculate_iou(mask_gt,mask_gen))

                # --- VISUALIZACIÓN ---
                if len(all_metrics["mae"]) <= args.num_samples_to_visualize:
                    row = test_df.iloc[global_idx]
                    s_id,p_id = str(row['sample_id']), str(row['path_number'])
                    
                    path_coords = []
                    if paths_coords_b and len(paths_coords_b) == 2 and paths_coords_b[0][i].numel() > 0:
                        x_coords_sample, y_coords_sample = paths_coords_b[0][i].tolist(), paths_coords_b[1][i].tolist()
                        path_coords = list(zip(x_coords_sample, y_coords_sample))
                    
                    rx_np_obs = real_X_b[i,0].cpu().numpy()
                    fig,axs = plt.subplots(1,3,figsize=(18,6))
                    fig.suptitle(f"Test: {s_id}_p{p_id}",fontsize=12)
                    
                    disp_in = np.stack([rx_np_obs]*3,axis=-1)
                    if path_coords and len(path_coords) > 1:
                        pts=np.array(path_coords,dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(disp_in,[pts],False,(0,1,0),1)
                        if len(pts)>0:
                            cv2.circle(disp_in,tuple(pts[0]),3,(0,0,1),-1)
                            cv2.circle(disp_in,tuple(pts[-1]),3,(1,0,0),-1)
                            
                    axs[0].imshow(np.clip(disp_in,0,1)); axs[0].set_title("Input(Obst+Path)"); axs[0].axis('off')
                    im1=axs[1].imshow(fy_np,cmap='viridis',vmin=0,vmax=1); axs[1].set_title("Generated"); axs[1].axis('off'); fig.colorbar(im1,ax=axs[1])
                    im2=axs[2].imshow(ry_np,cmap='viridis',vmin=0,vmax=1); axs[2].set_title("Ground Truth"); axs[2].axis('off'); fig.colorbar(im2,ax=axs[2])
                    
                    plt.tight_layout(rect=[0,0.03,1,0.93])
                    plt.savefig(os.path.join(EVAL_IMAGES_DIR,f"test_s{s_id}_p{p_id}.png"))
                    plt.close(fig)

    # --- CÁLCULO Y GUARDADO DE MÉTRICAS FINALES ---
    final_metrics = {"training_run_name":args.training_run_name, "num_test_samples":len(test_df)}
    for key,values in all_metrics.items():
        valid_v = [v for v in values if not np.isnan(v)]
        final_metrics[f"{key}_mean"] = float(np.mean(valid_v)) if valid_v else np.nan
        final_metrics[f"{key}_std"] = float(np.std(valid_v)) if valid_v else np.nan
    final_metrics["iou_threshold_used"] = args.iou_threshold
    
    print("\n--- Resultados Agregados de Evaluación ---")
    print(json.dumps(final_metrics,indent=4))
    
    with open(METRICS_JSON_PATH,'w') as f:
        json.dump(final_metrics,f,indent=4)
    print(f"Métricas JSON guardadas en: {METRICS_JSON_PATH}")

if __name__ == '__main__':
    cli_args = get_args_predict()
    evaluate_model_main(cli_args)