# TFM Notes

## <ins> 21/07/2025 </ins>

### Plan de Acción: Evolución del Proyecto de Mapeo de Gas

Este documento detalla los pasos necesarios para mejorar el sistema de predicción de mapas de difusión de gas.

#### Fase 1: Simulación Avanzada - Implementación de Viento Localizado

**Objetivo:** Aumentar el realismo de los datos de entrenamiento (Ground Truth) incorporando un efecto de advección (viento) que se origina en un punto y se propaga en forma de cono.

##### 1.1. Modificar el Generador de Difusión (`diffusion_generator.py`)

*   **[ ] Crear Nueva Función: `create_localized_wind_field()`**
    *   **Entradas:**
        *   Dimensiones del mapa: `height`, `width`
        *   Posición de origen del viento: `wind_source_pos` (tupla `i, j`)
        *   Dirección principal del viento: `wind_direction_vector` (tupla `vy, vx`)
        *   Fuerza máxima del viento: `wind_max_strength`
        *   Ángulo de apertura del cono: `cone_angle_deg`
        *   Potencia de atenuación: `falloff_power`
    *   **Lógica:** Generar un campo de vectores 2D que modele el cono de viento, atenuándose con la distancia.
    *   **Salida:** Dos arrays `(wind_field_vy, wind_field_vx)`.

*   **[ ] Modificar la Función de Simulación Principal (`generate_diffusion_map_roi`)**
    *   **Nuevas Entradas:** Añadir los parámetros del cono de viento.
    *   **Lógica Interna:** Precalcular el campo de viento llamando a `create_localized_wind_field()` y usarlo en el bucle de simulación para calcular el término de advección.
    *   **Nueva Salida:** La función deberá devolver 4 arrays: `(final_map, obstacle_mask, wind_field_vy, wind_field_vx)`.

##### 1.2. Actualizar el Generador de Dataset (`full_data_gen.py`)

*   **[ ] Añadir Nuevos Parámetros de Configuración:** Definir rangos para la generación aleatoria de los parámetros del cono de viento.
*   **[ ] Modificar el Bucle de Generación:** Para cada `sample_id`, generar aleatoriamente los parámetros del cono de viento y pasarlos a la función de simulación.
*   **[ ] Extender el Guardado de Datos:** Por cada `sample_id`, guardar los 4 arrays devueltos como archivos `.npy` separados:
    *   `sample_XXXXX_gt.npy`
    *   `sample_XXXXX_obstacles.npy`
    *   `sample_XXXXX_wind_vy.npy`
    *   `sample_XXXXX_wind_vx.npy`
*   **[ ] Actualizar `metadata.csv`:** Añadir nuevas columnas para registrar todos los parámetros del cono de viento y los nombres de los nuevos archivos.

---

#### Fase 2: Adaptación del Pipeline de Datos y Modelo

**Objetivo:** Asegurar que la nueva información sobre el viento se procese correctamente y se entregue como entrada al modelo de deep learning.

##### 2.1. Crear/Modificar el Script de Preprocesamiento (`preprocessor.py`)

*   **[ ] Actualizar la Lógica de Carga:** El script debe leer los dos nuevos mapas de viento (`_wind_vy.npy`, `_wind_vx.npy`).
*   **[ ] Modificar la Creación de la Entrada del Modelo:**
    *   El tensor de entrada (`_input.npy`) ahora tendrá **5 canales** en el formato `(H, W, 5)`:
        1.  **Canal 0:** Mapa de obstáculos.
        2.  **Canal 1:** Máscara de la trayectoria del robot (con grosor, ej. 3px).
        3.  **Canal 2:** Valores de detección de **gas** en la trayectoria.
        4.  **Canal 3:** Mapa del campo de viento `vy`.
        5.  **Canal 4:** Mapa del campo de viento `vx`.

##### 2.2. Modificar la Arquitectura del Modelo (`models.py`)

*   **[ ] `UNetGenerator`:**
    *   Cambiar la primera capa convolucional para que acepte `in_channels=5`.
*   **[ ] `PatchDiscriminator`:**
    *   Cambiar su primera capa convolucional para que acepte `in_channels = 5 (condición) + 1 (salida) = 6`.

##### 2.3. Simplificar `dataset.py`

*   **[ ] Mantener la versión simple:** El método `__getitem__` debe cargar el `.npy` de entrada (ahora con 5 canales) y el `.npy` de salida, y devolver **solo esos dos tensores**.

---

#### Fase 3: Flujo de Experimentación Automatizado

**Objetivo:** Crear un sistema robusto para probar diferentes configuraciones, visualizar resultados y encontrar el mejor modelo.

##### 3.1. Parametrizar `train.py` y `predict.py` con `argparse`

*   **[ ] `train.py`:**
    *   Convertir todos los hiperparámetros (LR, `lambda_l1`, `gen_features`, etc.) y rutas en argumentos de línea de comandos.
    *   **Visualización "Just-in-Time":** En el bucle de validación, cargar el archivo CSV de la ruta bajo demanda para dibujar la trayectoria sobre la imagen de entrada en los plots de muestra.
*   **[ ] `predict.py`:**
    *   Parametrizar para aceptar un `training_run_name` como argumento principal.
    *   **Salida JSON:** Guardar un resumen completo de todas las métricas en un archivo `test_metrics_summary.json`.
    *   Implementar la misma lógica de visualización "Just-in-Time" que en `train.py`.

##### 3.2. Crear el Script Orquestador (`run_hyperparam_tests.py`)

*   **[ ] Ubicación:** `TFM/src/gan/run_hyperparam_tests.py`.
*   **[ ] Definir el Espacio de Búsqueda:**
    *   Crear un diccionario `param_space` con los hiperparámetros a variar.
    *   Añadir un interruptor `SEARCH_MODE` para elegir entre `"RANDOM"` y `"GRID"`.
*   **[ ] Lógica de Generación de Configuraciones:**
    *   Implementar una función que genere `N` configuraciones aleatorias (sin duplicados) o todas las combinaciones de la rejilla.
*   **[ ] Bucle de Orquestación:**
    *   Iterar sobre cada configuración.
    *   Crear un directorio de salida único en `models_outputs/random_search/`.
    *   Usar `subprocess.run()` para ejecutar `train.py` y luego `predict.py` con los argumentos correspondientes.
*   **[ ] Recopilación y Resumen de Resultados:**
    *   Leer el `.json` de métricas de cada prueba.
    *   Guardar los resultados acumulados en un archivo `.jsonl`.
    *   Al final, convertir el `.jsonl` en un único `hyperparam_search_summary.csv` ordenado por la métrica de rendimiento principal (ej. `iou_mean`).