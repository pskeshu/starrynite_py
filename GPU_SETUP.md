# GPU Setup Guide for StarryNite

## Current System
- GPU: NVIDIA RTX A5000 (24GB VRAM)
- Driver: 581.15 (supports CUDA 13.0)
- Installed CUDA Toolkit: 10.0 (too old)
- Python: 3.11 (TF 2.10 needs 3.10)

## Problem
TensorFlow dropped native Windows GPU support after v2.10. TF 2.10 only has wheels for Python 3.7-3.10.

## Solution: Python 3.10 + TF 2.10 + CUDA 11.2

### Step 1: Install Python 3.10
Download from: https://www.python.org/downloads/release/python-31011/
Install to: `C:\Users\christensenr\AppData\Local\Programs\Python\Python310\`

### Step 2: Install CUDA 11.2 + cuDNN 8.1
- CUDA 11.2: https://developer.nvidia.com/cuda-11.2.0-download-archive
- cuDNN 8.1 for CUDA 11.2: https://developer.nvidia.com/rdp/cudnn-archive
  - Extract cuDNN to CUDA install dir (copy bin/, include/, lib/ contents)
- Your driver (581.15) is backward compatible with CUDA 11.2

### Step 3: Create GPU virtual environment
```bash
"C:\Users\christensenr\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv-gpu
source venv-gpu/Scripts/activate  # or venv-gpu\Scripts\activate on cmd
pip install tensorflow==2.10.1
pip install stardist
pip install -e ".[dev]"
```

### Step 4: Verify GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Expected Speedup
- CPU (current): ~29s per volume (240x244x410)
- GPU RTX A5000: ~2-5s per volume (estimated 6-15x speedup)
- Full dataset (1150 timepoints): CPU ~9.3 hours → GPU ~1-1.5 hours

## Alternative: WSL2
If CUDA 11.2 setup is problematic:
```bash
wsl --install -d Ubuntu-22.04
# Then in WSL2:
pip install tensorflow stardist
# GPU works automatically with modern TF in WSL2
```
