#!/bin/bash

# Script wrapper para ejecutar create_path_comparison.py con las librerías CUDA correctas
# Este script configura automáticamente LD_LIBRARY_PATH para que CuPy pueda usar GPU

# Obtener el directorio del script actual
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Buscar las librerías CUDA en el entorno Python
PYTHON_ENV_ROOT=$(python -c "import sys; print(sys.prefix)")
CUDA_NVRTC_PATH="${PYTHON_ENV_ROOT}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_RUNTIME_PATH="${PYTHON_ENV_ROOT}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

# Configurar LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CUDA_NVRTC_PATH}:${CUDA_RUNTIME_PATH}:$LD_LIBRARY_PATH"

echo "🔧 Configurando librerías CUDA..."
echo "   CUDA NVRTC: $CUDA_NVRTC_PATH"
echo "   CUDA Runtime: $CUDA_RUNTIME_PATH"
echo ""

# Cambiar al directorio del script y ejecutar
cd "$SCRIPT_DIR"
python create_path_comparison.py "$@"
