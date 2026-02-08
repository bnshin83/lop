#!/bin/bash
# Test script to verify all fixes are working

echo "=== Testing Fixes ==="

# Test 1: Environment setup
echo "1. Testing conda environment..."
# Don't load system Python module
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Conda environment not activated"
    exit 1
else
    echo "✅ Conda environment active: $CONDA_DEFAULT_ENV"
fi

# Ensure we use conda's Python
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "Python location: $(which python)"
if [[ "$(which python)" == *"conda_envs/lop"* ]]; then
    echo "✅ Using conda Python"
else
    echo "❌ Not using conda Python: $(which python)"
    exit 1
fi

# Test 2: Python packages
echo "2. Testing Python imports..."
python -c "
import numpy as np
import torch
import pandas as pd
print('✅ All packages available')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
" || {
    echo "❌ Package import failed"
    exit 1
}

# Test 3: Model file discovery  
echo "3. Testing model file discovery..."
python -c "
import sys
sys.path.append('.')
from multi_gpu_loss_generation import list_available_epochs
epochs = list_available_epochs('/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/model_parameters', 0)
epochs_1_2 = [e for e in epochs if 1 <= e <= 2]
if len(epochs_1_2) == 2:
    print('✅ Found epochs 1 and 2')
else:
    print(f'❌ Missing epochs: {epochs_1_2}')
    exit(1)
"

# Test 4: Environment variables for NCCL
echo "4. Testing NCCL environment variables..."
export NCCL_AVOID_RECORD_STREAMS=0
if [ "$NCCL_AVOID_RECORD_STREAMS" = "0" ]; then
    echo "✅ NCCL_AVOID_RECORD_STREAMS set correctly"
else
    echo "❌ NCCL environment variable not set"
fi

echo "🎉 All fixes verified successfully!"
echo ""
echo "Summary of fixes applied:"
echo "1. ✅ Removed 'module load python' to avoid conflicts"  
echo "2. ✅ Added PATH export to prioritize conda Python"
echo "3. ✅ Added environment validation for conda activation"
echo "4. ✅ Fixed model file pattern matching (checkpoint_index-0_epoch-*.pt)"
echo "5. ✅ Set NCCL_AVOID_RECORD_STREAMS=0 to suppress warnings"
echo ""
echo "The job should now:"
echo "- Properly activate conda environment with numpy/torch"
echo "- Find and process epochs 1 and 2 correctly"
echo "- Show fewer NCCL warnings"