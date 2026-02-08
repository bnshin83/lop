#!/bin/bash
#SBATCH --job-name=save_sample_order
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_save_sample_order.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_save_sample_order.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# =============================================================================
# Sample Order Logging Experiment Configuration
# =============================================================================

# Set task to run (19 = all 100 classes, matching successful random experiments)
TASK_NUMBER=19

# Set number of epochs to capture sample orders for
EPOCHS_PER_TASK=200

# Set save format for sample orders
# Options: "npy" (array only), "npz" (array+metadata), "both" (npy and npz)
SAVE_FORMAT="npz"

# Set random seed (optional - if not set, will use run_index based seed)
# Use specific seed to replicate successful experiments, e.g., 37542 from analysis
RANDOM_SEED=37542

# Set scratch mode: "true" to reinitialize network, "false" for normal behavior
SCRATCH_MODE="true"

# Set GPU ID
GPU_ID=0

# Set run index
RUN_INDEX=0

# =============================================================================
# Directory Setup
# =============================================================================

BASE_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar"
RESULTS_DIR="${BASE_DIR}/results_sample_order_logging_task_${TASK_NUMBER}_epochs_${EPOCHS_PER_TASK}_seed_${RANDOM_SEED}"

# Create results directory
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/sample_orders

echo "Results will be saved to: ${RESULTS_DIR}"
echo "Sample orders will be saved in: ${RESULTS_DIR}/sample_orders"

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Verify environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# =============================================================================
# Experiment Execution
# =============================================================================

echo "==================================================================="
echo "SAMPLE ORDER LOGGING EXPERIMENT"
echo "==================================================================="
echo "Task: ${TASK_NUMBER} (all 100 classes)"
echo "Epochs: ${EPOCHS_PER_TASK}"
echo "Save Format: ${SAVE_FORMAT}"
echo "Random Seed: ${RANDOM_SEED}"
echo "Scratch Mode: ${SCRATCH_MODE}"
echo "GPU ID: ${GPU_ID}"
echo "Results Dir: ${RESULTS_DIR}"
echo "==================================================================="

# Build command arguments
CMD_ARGS="--config_file ./cfg/base_deep_learning_system.json"
CMD_ARGS="$CMD_ARGS --results_dir ${RESULTS_DIR}"
CMD_ARGS="$CMD_ARGS --run_index ${RUN_INDEX}"
CMD_ARGS="$CMD_ARGS --gpu_id ${GPU_ID}"
CMD_ARGS="$CMD_ARGS --epochs_per_task ${EPOCHS_PER_TASK}"
CMD_ARGS="$CMD_ARGS --start_task ${TASK_NUMBER}"
CMD_ARGS="$CMD_ARGS --save_format ${SAVE_FORMAT}"

# Add scratch mode if enabled
if [ "$SCRATCH_MODE" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --scratch"
    echo "SCRATCH MODE ENABLED: Network will be reinitialized"
fi

# Add random seed if specified
if [ -n "$RANDOM_SEED" ]; then
    CMD_ARGS="$CMD_ARGS --random_seed ${RANDOM_SEED}"
    echo "USING CUSTOM RANDOM SEED: ${RANDOM_SEED}"
fi

echo "Command: python3.8 save_sample_order_experiment.py $CMD_ARGS"
echo "==================================================================="

# Run the sample order logging experiment
python3.8 save_sample_order_experiment.py $CMD_ARGS

# =============================================================================
# Post-Processing and Verification
# =============================================================================

echo "==================================================================="
echo "EXPERIMENT COMPLETED - VERIFYING RESULTS"
echo "==================================================================="

# Check if sample orders were saved
SAMPLE_ORDERS_DIR="${RESULTS_DIR}/sample_orders"
if [ -d "$SAMPLE_ORDERS_DIR" ]; then
    echo "Sample orders directory exists: ${SAMPLE_ORDERS_DIR}"
    
    # Count saved files
    NPY_COUNT=$(find "$SAMPLE_ORDERS_DIR" -name "*.npy" | wc -l)
    NPZ_COUNT=$(find "$SAMPLE_ORDERS_DIR" -name "*.npz" | wc -l)
    JSON_COUNT=$(find "$SAMPLE_ORDERS_DIR" -name "*.json" | wc -l)
    
    echo "Files saved:"
    echo "  NPY files: ${NPY_COUNT}"
    echo "  NPZ files: ${NPZ_COUNT}"
    echo "  JSON files: ${JSON_COUNT}"
    
    # Show first few files
    echo ""
    echo "First 5 sample order files:"
    ls -la "$SAMPLE_ORDERS_DIR" | head -8
    
    # Show disk usage
    echo ""
    echo "Disk usage:"
    du -sh "$SAMPLE_ORDERS_DIR"
    
    # Verify a sample file can be loaded
    if [ "$NPZ_COUNT" -gt 0 ]; then
        echo ""
        echo "Testing sample order file loading..."
        python3.8 -c "
import numpy as np
import os
import glob

sample_dir = '${SAMPLE_ORDERS_DIR}'
npz_files = glob.glob(os.path.join(sample_dir, '*.npz'))

if npz_files:
    sample_file = npz_files[0]
    print(f'Testing file: {sample_file}')
    
    try:
        data = np.load(sample_file)
        print(f'Keys in file: {list(data.keys())}')
        
        if 'sample_order' in data:
            sample_order = data['sample_order']
            print(f'Sample order shape: {sample_order.shape}')
            print(f'Sample order dtype: {sample_order.dtype}')
            print(f'First 10 samples: {sample_order[:10]}')
            
        # Print metadata if available
        metadata_keys = [k for k in data.keys() if k != 'sample_order']
        if metadata_keys:
            print(f'Metadata keys: {metadata_keys}')
            for key in metadata_keys[:5]:  # Show first 5 metadata items
                print(f'  {key}: {data[key]}')
                
        print('✓ Sample order file loads successfully')
        
    except Exception as e:
        print(f'✗ Error loading sample order file: {e}')
else:
    print('No NPZ files found to test')
"
    fi
    
else
    echo "ERROR: Sample orders directory not found: ${SAMPLE_ORDERS_DIR}"
    exit 1
fi

echo "==================================================================="
echo "SAMPLE ORDER LOGGING EXPERIMENT COMPLETED SUCCESSFULLY"
echo ""
echo "USAGE INSTRUCTIONS:"
echo "1. Sample orders are saved in: ${SAMPLE_ORDERS_DIR}"
echo "2. Each epoch has its own file: task_${TASK_NUMBER}_epoch_XXXX_sample_order.npz"
echo "3. To load sample order for epoch N:"
echo "   import numpy as np"
echo "   data = np.load('task_${TASK_NUMBER}_epoch_{N:04d}_sample_order.npz')"
echo "   sample_order = data['sample_order']  # Array of sample indices"
echo "   metadata = {k: data[k] for k in data.keys() if k != 'sample_order'}"
echo ""
echo "4. To replicate this exact random ordering in a new experiment:"
echo "   - Use random_seed=${RANDOM_SEED}"
echo "   - Use the saved sample orders as predetermined ordering"
echo "   - Match the experimental setup (task ${TASK_NUMBER}, ${EPOCHS_PER_TASK} epochs)"
echo "==================================================================="
