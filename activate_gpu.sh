#!/bin/bash
# Activate GPU venv with CUDA libraries on PATH
source venv-gpu/Scripts/activate
NVIDIA_BASE="$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))")"
export PATH="$NVIDIA_BASE/cuda_runtime/bin:$NVIDIA_BASE/cublas/bin:$NVIDIA_BASE/cudnn/bin:$NVIDIA_BASE/cufft/bin:$NVIDIA_BASE/curand/bin:$NVIDIA_BASE/cusolver/bin:$NVIDIA_BASE/cusparse/bin:$PATH"
echo "GPU venv activated with CUDA on PATH"
python -c "import tensorflow as tf; print('TF:', tf.__version__); gpus=tf.config.list_physical_devices('GPU'); print('GPU:', gpus[0].name if gpus else 'None')"
