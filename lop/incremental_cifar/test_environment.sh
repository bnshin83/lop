#!/bin/bash
# Test script to verify conda environment setup

echo "=== Testing Conda Environment Setup ==="
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"

# Load conda
echo "Loading conda..."
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh

# Activate environment
echo "Activating conda environment..."
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Verify environment
echo "=== Environment Verification ==="
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Test critical imports
echo "=== Testing Package Imports ==="
$CONDA_PREFIX/bin/python -c "
try:
    import numpy as np
    import torch
    import pandas as pd
    import tqdm
    print('✅ All critical packages available')
    print(f'NumPy version: {np.__version__}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Pandas version: {pd.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Environment test passed!"
else
    echo "❌ Environment test failed!"
    exit 1
fi

echo "=== Environment is ready for multi-GPU jobs ==="
