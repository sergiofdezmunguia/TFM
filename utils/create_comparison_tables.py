import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN ---
CSV_NO_WIND_PATH = '/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparam_search/search_summary.csv'
CSV_WITH_WIND_PATH = '/home/sergio/uni/master/tfm/TFM/models_outputs/hyperparams_search_wind/search_summary.csv'
OUTPUT_DIR = '/home/sergio/uni/master/tfm/tfm_memoria'
TABLE1_FILE = 'table1_no_wind.tex'
TABLE2_FILE = 'table2_with_wind.tex' 
TABLE3_FILE = 'table3_comparison.tex'
TOP_N_RUNS = 5
SORTING_METRIC = 'iou_mean'
SORTING_ASCENDING = False

def format_and_clean_df(filepath):
    """Carga, limpia y ordena el DataFrame de resultados."""
    if not os.path.exists(filepath):
        print(f"ERROR: Archivo no encontrado en {filepath}")
        return None
    df = pd.read_csv(filepath)
    metric_cols = ['iou_mean', 'peak_dist_mean', 'peak_int_err_mean', 'mae_mean', 'psnr_mean', 'ssim_mean', 
                   'iou_std', 'peak_dist_std', 'peak_int_err_std', 'mae_std', 'psnr_std', 'ssim_std']
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=[SORTING_METRIC], inplace=True)
    df = df.sort_values(by=SORTING_METRIC, ascending=SORTING_ASCENDING).reset_index(drop=True)
    return df

def generate_latex_table_detailed(df, caption, label):
    """Genera el código LaTeX para una tabla detallada usando el formato personalizado."""
    df_display = df.copy()
    df_display = df_display[['run_name', 'lr_g', 'lambda_l1', 'gen_features', 
                             'iou_mean', 'peak_dist_mean', 'psnr_mean', 'ssim_mean']]
    
    # Shorten run names to just the trial number
    df_display['run_short'] = df_display['run_name'].str.extract(r'(\d+)').astype(int)
    df_display['lr_g'] = df_display['lr_g'].map('{:.1e}'.format)
    df_display['lambda_l1'] = df_display['lambda_l1'].map('{:.0f}'.format)
    df_display['iou_mean'] = df_display['iou_mean'].map('{:.3f}'.format)
    df_display['peak_dist_mean'] = df_display['peak_dist_mean'].map('{:.1f}'.format)
    df_display['psnr_mean'] = df_display['psnr_mean'].map('{:.2f}'.format)
    df_display['ssim_mean'] = df_display['ssim_mean'].map('{:.3f}'.format)

    # Generate table rows
    table_rows = []
    for _, row in df_display.iterrows():
        row_str = f"    {row['run_short']:02d} & {row['lr_g']} & {row['lambda_l1']} & {row['gen_features']} & {row['iou_mean']} & {row['peak_dist_mean']} & {row['psnr_mean']} & {row['ssim_mean']} \\\\"
        table_rows.append(row_str)
    
    full_table_str = f"""\\begin{{table}}[{caption}]{{{label}}}{{{caption}}}
  \\begin{{tabular}}{{cccccccc}}
    \\hline
    \\textbf{{ID}} & \\textbf{{LR}} & \\textbf{{$\\lambda_{{L1}}$}} & \\textbf{{Gen}} & \\textbf{{IoU}} & \\textbf{{Peak}} & \\textbf{{PSNR}} & \\textbf{{SSIM}} \\\\
    \\hline \\hline
{chr(10).join(table_rows)}
    \\hline
  \\end{{tabular}}
\\end{{table}}
"""
    return full_table_str

def generate_latex_table_summary(best_no_wind, best_with_wind, caption, label):
    """Genera el código LaTeX para la tabla resumen usando el formato personalizado."""
    data_rows = [
        ('IoU', 
         f"{best_no_wind['iou_mean']:.3f}",
         f"{best_with_wind['iou_mean']:.3f}"),
        ('Peak Dist', 
         f"{best_no_wind['peak_dist_mean']:.1f}",
         f"{best_with_wind['peak_dist_mean']:.1f}"),
        ('Peak Err',
         f"{best_no_wind['peak_int_err_mean']:.3f}",
         f"{best_with_wind['peak_int_err_mean']:.3f}"),
        ('MAE', 
         f"{best_no_wind['mae_mean']:.4f}",
         f"{best_with_wind['mae_mean']:.4f}"),
        ('PSNR', 
         f"{best_no_wind['psnr_mean']:.2f}",
         f"{best_with_wind['psnr_mean']:.2f}"),
        ('SSIM', 
         f"{best_no_wind['ssim_mean']:.3f}",
         f"{best_with_wind['ssim_mean']:.3f}")
    ]
    
    table_rows = []
    for metric, no_wind, with_wind in data_rows:
        table_rows.append(f"    {metric} & {no_wind} & {with_wind} \\\\")
    
    full_table_str = f"""\\begin{{table}}[{caption}]{{{label}}}{{{caption}}}
  \\begin{{tabular}}{{lcc}}
    \\hline
    \\textbf{{Métrica}} & \\textbf{{Sin Viento}} & \\textbf{{Con Viento}} \\\\
    \\hline \\hline
{chr(10).join(table_rows)}
    \\hline
  \\end{{tabular}}
\\end{{table}}
"""
    return full_table_str

if __name__ == '__main__':
    # Cargar y procesar ambos dataframes
    df_no_wind = format_and_clean_df(CSV_NO_WIND_PATH)
    df_with_wind = format_and_clean_df(CSV_WITH_WIND_PATH)

    # Crear el directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generar tabla 1: Sin viento
    if df_no_wind is not None:
        table1_path = os.path.join(OUTPUT_DIR, TABLE1_FILE)
        with open(table1_path, 'w') as f:
            f.write("%% ============================================================= %%\n")
            f.write(f"%% Tabla generada automáticamente por create_comparison_tables.py %%\n")
            f.write("%% NO EDITAR MANUALMENTE. %%\n")
            f.write("%% ============================================================= %%\n\n")
            
            table_no_wind_latex = generate_latex_table_detailed(
                df_no_wind.head(TOP_N_RUNS),
                caption="Top 5 modelos dataset sin viento",
                label="tab:hpt_no_wind"
            )
            f.write(table_no_wind_latex)
        print(f"Tabla 1 generada: {table1_path}")

    # Generar tabla 2: Con viento
    if df_with_wind is not None:
        table2_path = os.path.join(OUTPUT_DIR, TABLE2_FILE)
        with open(table2_path, 'w') as f:
            f.write("%% ============================================================= %%\n")
            f.write(f"%% Tabla generada automáticamente por create_comparison_tables.py %%\n")
            f.write("%% NO EDITAR MANUALMENTE. %%\n")
            f.write("%% ============================================================= %%\n\n")
            
            table_with_wind_latex = generate_latex_table_detailed(
                df_with_wind.head(TOP_N_RUNS),
                caption="Top 5 modelos dataset con viento",
                label="tab:hpt_with_wind"
            )
            f.write(table_with_wind_latex)
        print(f"Tabla 2 generada: {table2_path}")

    # Generar tabla 3: Comparación
    if df_no_wind is not None and df_with_wind is not None:
        table3_path = os.path.join(OUTPUT_DIR, TABLE3_FILE)
        with open(table3_path, 'w') as f:
            f.write("%% ============================================================= %%\n")
            f.write(f"%% Tabla generada automáticamente por create_comparison_tables.py %%\n")
            f.write("%% NO EDITAR MANUALMENTE. %%\n")
            f.write("%% ============================================================= %%\n\n")
            
            best_no_wind_row = df_no_wind.iloc[0]
            best_with_wind_row = df_with_wind.iloc[0]
            summary_table_latex = generate_latex_table_summary(
                best_no_wind_row,
                best_with_wind_row,
                caption="Comparativa mejores modelos",
                label="tab:summary_comparison"
            )
            f.write(summary_table_latex)
        print(f"Tabla 3 generada: {table3_path}")

    print(f"¡Éxito! Las 3 tablas de LaTeX se han guardado en: {OUTPUT_DIR}")
    print("Archivos generados:")
    print(f"  - {TABLE1_FILE} (Sin viento)")
    print(f"  - {TABLE2_FILE} (Con viento)")
    print(f"  - {TABLE3_FILE} (Comparación)")
    print("Puedes incluir cada tabla individualmente en tu documento LaTeX usando \\input{filename.tex}")