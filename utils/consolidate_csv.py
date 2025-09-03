# consolidate_results.py
import os
import json
import pandas as pd

# --- CONFIGURACIÓN ---
# Apunta al directorio base donde se guardaron todas las corridas de la búsqueda
SEARCH_BASE_DIR = os.path.expanduser("~/uni/master/tfm/TFM/models_outputs/hyperparams_search_wind")
# Archivo de log que contiene la configuración de cada corrida
RESULTS_LOG_JSONL_FILE = os.path.join(SEARCH_BASE_DIR, "search_results.jsonl")
# Nombre del archivo CSV de salida
FINAL_SUMMARY_CSV_FILE = os.path.join(SEARCH_BASE_DIR, "search_summary.csv")

def consolidate_results():
    print(f"Buscando resultados en: {SEARCH_BASE_DIR}")
    print(f"Leyendo configuraciones desde: {RESULTS_LOG_JSONL_FILE}")

    if not os.path.exists(RESULTS_LOG_JSONL_FILE):
        print(f"ERROR: El archivo de log '{RESULTS_LOG_JSONL_FILE}' no existe. No se puede consolidar.")
        return

    all_results_data = []

    # Leer el archivo de log para obtener la configuración de cada corrida
    with open(RESULTS_LOG_JSONL_FILE, "r") as f_log:
        for line in f_log:
            if not line.strip(): continue
            try:
                log_entry = json.loads(line)
                run_name = log_entry.get("run_name")
                
                if not run_name: continue

                # Empezar a construir la fila para el CSV con la info del log
                result_row = {"run_name": run_name}
                result_row.update(log_entry.get("config_params", {})) # Añadir hiperparámetros
                result_row["train_status"] = log_entry.get("train_status")
                result_row["predict_status"] = log_entry.get("predict_status")

                # Si la predicción fue completada, buscar y cargar el archivo de métricas
                if log_entry.get("predict_status") == "METRICS_NOT_FOUND":
                    # Construir la ruta al archivo JSON de métricas
                    # El sufijo 'eval' puede cambiar si lo modificaste en predict.py
                    metrics_path = os.path.join(SEARCH_BASE_DIR, run_name, "evaluation_test_eval", "test_metrics_summary.json")
                    
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f_metrics:
                            eval_metrics = json.load(f_metrics)
                        # Añadir las métricas a la fila, excluyendo claves duplicadas
                        for key, value in eval_metrics.items():
                            if key not in result_row:
                                result_row[key] = value
                    else:
                        print(f"ADVERTENCIA: Métricas JSON no encontradas para la corrida completada '{run_name}' en '{metrics_path}'")

                all_results_data.append(result_row)

            except json.JSONDecodeError:
                print(f"ADVERTENCIA: Línea corrupta en log: {line.strip()}")

    if not all_results_data:
        print("No se encontraron datos de resultados para consolidar.")
        return

    # Crear el DataFrame y guardarlo como CSV
    summary_df = pd.DataFrame(all_results_data)
    
    # Ordenar por una métrica clave para ver los mejores resultados primero
    if "iou_mean" in summary_df.columns:
        summary_df = summary_df.sort_values(by="iou_mean", ascending=False)

    try:
        summary_df.to_csv(FINAL_SUMMARY_CSV_FILE, index=False)
        print(f"\n¡ÉXITO! Resumen consolidado con {len(summary_df)} corridas guardado en:")
        print(FINAL_SUMMARY_CSV_FILE)
    except Exception as e:
        print(f"\nERROR al guardar el archivo CSV: {e}")

if __name__ == "__main__":
    consolidate_results()