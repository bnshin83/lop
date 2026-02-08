#!/bin/bash
#SBATCH --job-name=multi_gpu_dormant
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_dormant.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_dormant.err

echo "=== Multi-GPU Per-Sample Dormant Analysis Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"

# Change to the submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load CUDA module (but not Python - we'll use conda's Python)
module load cuda

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi
echo "========================"

# Set environment variables for single-node multi-GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29503
export WORLD_SIZE=8
export NCCL_DEBUG=WARN
export NCCL_AVOID_RECORD_STREAMS=0

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT" 
echo "World size: $WORLD_SIZE"

# Set environment variable for output directory (matching incremental_cifar.sh)
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Ensure conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

# Ensure we use conda's Python by putting it first in PATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verify environment activation
echo "=== Verifying Environment ==="
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Test critical imports
python -c "
try:
    import numpy as np
    import torch
    import pandas as pd
    print('✅ All critical packages available')
    print(f'NumPy version: {np.__version__}')
    print(f'PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    exit(1)
" || {
    echo "❌ Environment setup failed - missing dependencies"
    exit 1
}

# Set OpenMP threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify paths exist
echo "=== Verifying Paths ==="
DATA_PATH="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data/cifar-100-python"
MODEL_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/model_parameters"
SAVE_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_dormant_analysis"
CLASS_ORDER_FILE="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.npy"

echo "Data path: $DATA_PATH"
echo "Model directory: $MODEL_DIR"
echo "Save directory: $SAVE_DIR"
echo "Class order file: $CLASS_ORDER_FILE"

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$CLASS_ORDER_FILE" ]; then
    echo "ERROR: Class order file does not exist: $CLASS_ORDER_FILE"
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

# Function to run the dormant analysis with error handling
run_dormant_analysis() {
    local start_epoch=$1
    local end_epoch=$2
    local threshold=$3
    
    echo "=== Starting Per-Sample Dormant Analysis ==="
    echo "Epoch range: $start_epoch to $end_epoch"
    echo "Dormant unit threshold: $threshold"
    echo "Start time: $(date)"
    
    # Run the multi-GPU dormant analysis script using torchrun for proper DDP setup
    torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        multi_gpu_dormant_analysis.py \
        --start_epoch $start_epoch \
        --end_epoch $end_epoch \
        --data_path "$DATA_PATH" \
        --model_dir "$MODEL_DIR" \
        --save_base_dir "$SAVE_DIR" \
        --class_order_file "$CLASS_ORDER_FILE" \
        --experiment_index 0 \
        --dormant_unit_threshold $threshold
    
    local exit_code=$?
    echo "Dormant Analysis completed with exit code: $exit_code"
    echo "End time: $(date)"
    
    return $exit_code
}

# Function to check progress and resume if needed
check_and_resume() {
    local start_epoch=$1
    local end_epoch=$2
    
    echo "=== Checking Progress ==="
    local completed_epochs=0
    local total_epochs=$(( end_epoch - start_epoch + 1 ))
    
    # Check for completed epochs
    for epoch in $(seq $start_epoch $end_epoch); do
        local epoch_dir="$SAVE_DIR/epoch_$(printf '%04d' $epoch)"
        if [ -f "$epoch_dir/per_sample_dormant_props.npy" ] && [ -f "$epoch_dir/combined_dormant_data.csv" ]; then
            ((completed_epochs++))
        fi
    done
    
    echo "Progress: $completed_epochs/$total_epochs epochs completed"
    
    if [ $completed_epochs -eq $total_epochs ]; then
        echo "All epochs already completed!"
        return 0
    else
        echo "Resuming analysis... ($((total_epochs - completed_epochs)) epochs remaining)"
        return 1
    fi
}

# Function to cleanup incomplete results
cleanup_incomplete_results() {
    echo "=== Cleaning up incomplete results ==="
    
    # Remove incomplete epoch directories
    find "$SAVE_DIR" -type d -name "epoch_*" -exec sh -c '
        epoch_dir="$1"
        if [ ! -f "$epoch_dir/per_sample_dormant_props.npy" ] || [ ! -f "$epoch_dir/combined_dormant_data.csv" ]; then
            echo "Removing incomplete epoch directory: $epoch_dir"
            rm -rf "$epoch_dir"
        fi
    ' _ {} \;
    
    echo "Cleanup completed."
}

# Function to validate results
validate_results() {
    local start_epoch=$1
    local end_epoch=$2
    
    echo "=== Validating Results ==="
    local valid_epochs=0
    local total_epochs=$(( end_epoch - start_epoch + 1 ))
    
    for epoch in $(seq $start_epoch $end_epoch); do
        local epoch_dir="$SAVE_DIR/epoch_$(printf '%04d' $epoch)"
        local required_files=(
            "per_sample_dormant_props.npy"
            "per_sample_dormant_props.csv"
            "per_sample_labels.npy"
            "per_sample_sample_ids.npy"
            "combined_dormant_data.csv"
            "metadata.json"
        )
        
        local all_files_exist=true
        for file in "${required_files[@]}"; do
            if [ ! -f "$epoch_dir/$file" ]; then
                all_files_exist=false
                break
            fi
        done
        
        if [ "$all_files_exist" = true ]; then
            ((valid_epochs++))
        else
            echo "❌ Incomplete results for epoch $epoch"
        fi
    done
    
    echo "Validation: $valid_epochs/$total_epochs epochs have complete results"
    
    if [ $valid_epochs -eq $total_epochs ]; then
        echo "✅ All epochs have complete results!"
        return 0
    else
        echo "⚠️  Some epochs have incomplete results"
        return 1
    fi
}

# Main execution - process epochs with detailed analysis
START_EPOCH=0
END_EPOCH=4000
STEP=1
DORMANT_THRESHOLD=0.01

echo "=== Job Configuration ==="
echo "Start epoch: $START_EPOCH"
echo "End epoch: $END_EPOCH"
echo "Dormant unit threshold: $DORMANT_THRESHOLD"
echo "Total epochs to process: $((END_EPOCH - START_EPOCH + 1))"
echo "========================="

# Check if we need to resume or start fresh
if check_and_resume $START_EPOCH $END_EPOCH; then
    echo "Job already completed successfully!"
    
    # Still run validation to confirm
    if validate_results $START_EPOCH $END_EPOCH; then
        echo "✅ All results validated successfully!"
    else
        echo "⚠️  Some results may be incomplete - consider re-running"
    fi
else
    # Clean up any incomplete results first
    cleanup_incomplete_results
    
    # Run the analysis
    if run_dormant_analysis $START_EPOCH $END_EPOCH $DORMANT_THRESHOLD; then
        echo "=== Per-Sample Dormant Analysis Completed Successfully ==="
        
        # Validate results
        if validate_results $START_EPOCH $END_EPOCH; then
            echo "✅ All results validated successfully!"
        else
            echo "⚠️  Some results may be incomplete"
        fi
        
        # Generate summary report
        echo "=== Generating Summary Report ==="
        python -c "
import os
import json
import numpy as np
from pathlib import Path

save_dir = '$SAVE_DIR'
epochs_processed = []
total_samples_per_epoch = []
total_time = 0
dormant_stats = {'means': [], 'stds': [], 'mins': [], 'maxs': []}

# Collect data from all epoch directories
for epoch_dir in sorted(Path(save_dir).glob('epoch_*')):
    metadata_file = epoch_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        epochs_processed.append(metadata['epoch'])
        total_samples_per_epoch.append(metadata['total_samples'])
        total_time += metadata['compute_time_seconds']
        
        # Collect dormant proportion statistics
        stats = metadata.get('dormant_props_stats', {})
        dormant_stats['means'].append(stats.get('mean', 0))
        dormant_stats['stds'].append(stats.get('std', 0))
        dormant_stats['mins'].append(stats.get('min', 0))
        dormant_stats['maxs'].append(stats.get('max', 0))

# Generate report
print('Per-Sample Dormant Analysis Summary Report:')
print('=' * 60)
print(f'Job Configuration:')
print(f'  Start epoch: $START_EPOCH')
print(f'  End epoch: $END_EPOCH')
print(f'  Step size: $STEP')
print(f'  Dormant threshold: $DORMANT_THRESHOLD')
print()
print(f'Results Summary:')
print(f'  Epochs processed: {len(epochs_processed)}')
if total_samples_per_epoch:
    print(f'  Samples per epoch: {total_samples_per_epoch[0]:,}')
    print(f'  Total sample-epoch pairs: {sum(total_samples_per_epoch):,}')
print(f'  Total computation time: {total_time/3600:.2f} hours')
if epochs_processed:
    print(f'  Average time per epoch: {total_time/len(epochs_processed)/60:.2f} minutes')
    if total_samples_per_epoch:
        throughput = sum(total_samples_per_epoch) / total_time
        print(f'  Throughput: {throughput:.1f} samples/second')

# Storage analysis
storage_gb = 0
file_counts = {'npy': 0, 'csv': 0, 'json': 0}

for ext in file_counts.keys():
    files = list(Path(save_dir).rglob(f'*.{ext}'))
    file_counts[ext] = len(files)
    storage_gb += sum(f.stat().st_size for f in files) / (1024**3)

print(f'\\nStorage Analysis:')
print(f'  .npy files: {file_counts[\"npy\"]}')
print(f'  .csv files: {file_counts[\"csv\"]}') 
print(f'  .json files: {file_counts[\"json\"]}')
print(f'  Total storage used: {storage_gb:.2f} GB')

# Dormant proportion statistics across epochs
if dormant_stats['means']:
    print(f'\\nDormant Proportion Statistics Across Epochs:')
    print(f'  Mean dormant prop (avg across epochs): {np.mean(dormant_stats[\"means\"]):.4f}')
    print(f'  Std dormant prop (avg across epochs): {np.mean(dormant_stats[\"stds\"]):.4f}')
    print(f'  Min dormant prop (global min): {min(dormant_stats[\"mins\"]):.4f}')
    print(f'  Max dormant prop (global max): {max(dormant_stats[\"maxs\"]):.4f}')

# Show epoch breakdown
if len(epochs_processed) <= 20:  # Show all if not too many
    print(f'\\nProcessed Epochs: {sorted(epochs_processed)}')
else:  # Show first few and last few
    sorted_epochs = sorted(epochs_processed)
    print(f'\\nProcessed Epochs: {sorted_epochs[:5]} ... {sorted_epochs[-5:]} ({len(epochs_processed)} total)')

print(f'\\n✅ Per-sample dormant analysis completed successfully!')
print(f'📁 Results location: {save_dir}')
"
        
        echo ""
        echo "=== Generated Files Structure ==="
        echo "Individual epoch files (per epoch):"
        echo "  - epoch_XXXX/per_sample_dormant_props.npy & .csv (dormant proportions per sample)"
        echo "  - epoch_XXXX/per_sample_sample_ids.npy & .csv (sample IDs)"
        echo "  - epoch_XXXX/per_sample_labels.npy & .csv (class labels)"
        echo "  - epoch_XXXX/combined_dormant_data.csv (all data combined)"
        echo "  - epoch_XXXX/metadata.json (epoch metadata)"
        echo ""
        echo "Key Features:"
        echo "  ✅ Per-sample dormant proportions (not batch-averaged)"
        echo "  ✅ Sample ID traceability across all epochs"
        echo "  ✅ Class-introduction aware processing"
        echo "  ✅ Multi-GPU distributed computation"
        echo "  ✅ Compatible with existing analysis pipeline"
        
    else
        echo "=== Per-Sample Dormant Analysis Failed ==="
        echo "Check error logs for details."
        exit 1
    fi
fi

echo "=== Job Finished ==="
echo "End time: $(date)"
echo "Check results in: $SAVE_DIR"