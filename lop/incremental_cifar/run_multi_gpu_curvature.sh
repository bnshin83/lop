#!/bin/bash
#SBATCH --job-name=multi_gpu_curvature
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_curvature.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_curvature.err

echo "=== Multi-GPU Curvature Analysis Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"

# Change to the submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi
echo "========================"

# Set environment variables for single-node multi-GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export NCCL_DEBUG=WARN

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT" 
echo "World size: $WORLD_SIZE"

# Set environment variable for output directory (matching incremental_cifar.sh)
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Set OpenMP threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify paths exist
echo "=== Verifying Paths ==="
DATA_PATH="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data/cifar-100-python"
MODEL_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/model_parameters"
SAVE_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_image_curvature_full"

echo "Data path: $DATA_PATH"
echo "Model directory: $MODEL_DIR"
echo "Save directory: $SAVE_DIR"

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"
echo "Created save directory: $SAVE_DIR"

# Count available model files
MODEL_COUNT=$(find "$MODEL_DIR" -name "checkpoint_index-0_epoch-*.pt" | wc -l)
echo "Found $MODEL_COUNT model files in $MODEL_DIR"

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No model files found in $MODEL_DIR"
    exit 1
fi

echo "========================"

# Function to run the analysis with error handling
run_curvature_analysis() {
    local start_epoch=$1
    local end_epoch=$2
    local step=$3
    
    echo "=== Starting Curvature Analysis ==="
    echo "Epoch range: $start_epoch to $end_epoch (step: $step)"
    echo "Start time: $(date)"
    
    # Run the multi-GPU script (single node, multiple GPUs)
    torchrun --nproc_per_node=8 multi_gpu_curvature_analysis.py \
        --start_epoch $start_epoch \
        --end_epoch $end_epoch \
        --step $step \
        --data_path "$DATA_PATH" \
        --model_dir "$MODEL_DIR" \
        --save_base_dir "$SAVE_DIR" \
        --experiment_index 0
    
    local exit_code=$?
    echo "Analysis completed with exit code: $exit_code"
    echo "End time: $(date)"
    
    return $exit_code
}

# Function to check progress and resume if needed
check_and_resume() {
    local start_epoch=$1
    local end_epoch=$2
    local step=$3
    
    echo "=== Checking Progress ==="
    local completed_epochs=0
    local total_epochs=$(( (end_epoch - start_epoch) / step + 1 ))
    # skip epochs
    for epoch in $(seq $start_epoch $step $end_epoch); do
        local epoch_dir="$SAVE_DIR/epoch_$(printf '%04d' $epoch)"
        if [ -f "$epoch_dir/per_image_curvature.npy" ]; then
            ((completed_epochs++))
        fi
    done
    # skip epochs
    echo "Progress: $completed_epochs/$total_epochs epochs completed"
    
    if [ $completed_epochs -eq $total_epochs ]; then
        echo "All epochs already completed!"
        return 0
    else
        echo "Resuming analysis..."
        return 1
    fi
}

# Main execution - process more epochs for detailed analysis
START_EPOCH=1
END_EPOCH=4000  
STEP=1

# Check if we need to resume or start fresh
if check_and_resume $START_EPOCH $END_EPOCH $STEP; then
    echo "Job already completed successfully!"
else
    # Run the analysis
    if run_curvature_analysis $START_EPOCH $END_EPOCH $STEP; then
        echo "=== Job Completed Successfully ==="
        
        # Generate summary report
        echo "=== Generating Summary Report ==="
        $CONDA_PREFIX/bin/python -c "
import os
import json
import numpy as np
from pathlib import Path

save_dir = '$SAVE_DIR'
epochs_processed = []
total_samples = 0
total_time = 0

for epoch_dir in sorted(Path(save_dir).glob('epoch_*')):
    metadata_file = epoch_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        epochs_processed.append(metadata['epoch'])
        total_samples = metadata['total_samples']  # Same for all epochs
        total_time += metadata['compute_time_seconds']

print(f'Summary Report:')
print(f'  Epochs processed: {len(epochs_processed)}')
print(f'  Samples per epoch: {total_samples:,}')
print(f'  Total computation time: {total_time/3600:.2f} hours')
print(f'  Average time per epoch: {total_time/len(epochs_processed)/60:.2f} minutes')
print(f'  Throughput: {total_samples * len(epochs_processed) / total_time:.1f} samples/second')
print(f'  Storage used: {sum(f.stat().st_size for f in Path(save_dir).rglob(\"*.npy\")) / (1024**3):.2f} GB')
"
        
    else
        echo "=== Job Failed ==="
        echo "Check error logs for details."
        exit 1
    fi
fi

echo "=== Job Finished ==="
echo "End time: $(date)"
echo "Check results in: $SAVE_DIR"
