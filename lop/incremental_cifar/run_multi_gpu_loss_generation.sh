#!/bin/bash
#SBATCH --job-name=multi_gpu_loss_gen
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_loss_gen.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_loss_gen.err

echo "=== Multi-GPU Loss/Grad Generation Job Started ==="
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
export MASTER_PORT=29502
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
MODEL_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/model_parameters"
DATA_PATH="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data"
OUTPUT_LOSS_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_full"
CLASS_ORDER_FILE="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.npy"

echo "Model directory: $MODEL_DIR"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_LOSS_DIR"
echo "Class order file: $CLASS_ORDER_FILE"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -f "$CLASS_ORDER_FILE" ]; then
    echo "ERROR: Class order file does not exist: $CLASS_ORDER_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_LOSS_DIR"
echo "Created output directory: $OUTPUT_LOSS_DIR"

# Count available model files
MODEL_COUNT=$(find "$MODEL_DIR" -name "index-0_epoch-*.pt" | wc -l)
echo "Found $MODEL_COUNT model files in $MODEL_DIR"

if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No model files found in $MODEL_DIR"
    exit 1
fi

echo "========================"

# Function to run the loss/grad generation with error handling
run_loss_generation() {
    local start_epoch=$1
    local end_epoch=$2
    
    echo "=== Starting Loss/Grad Generation ==="
    echo "Epoch range: $start_epoch to $end_epoch"
    echo "Start time: $(date)"
    
    # Run the multi-GPU loss generation script using torchrun for proper DDP setup
    torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        multi_gpu_loss_generation.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$OUTPUT_LOSS_DIR" \
        --data_path "$DATA_PATH" \
        --class_order_file "$CLASS_ORDER_FILE" \
        --experiment_index 0 \
        --start_epoch $start_epoch \
        --end_epoch $end_epoch \
        --batch_size 256 \
        --num_workers 4 \
        --compute_grad
    
    local exit_code=$?
    echo "Loss/Grad Generation completed with exit code: $exit_code"
    echo "End time: $(date)"
    
    return $exit_code
}

# Function to check if generation is already completed
check_generation_completed() {
    local start_epoch=$1
    local end_epoch=$2
    
    echo "=== Checking Generation Progress ==="
    local completed_epochs=0
    local total_epochs=$(( end_epoch - start_epoch + 1 ))
    
    # Check for individual epoch directories
    for epoch in $(seq $start_epoch $end_epoch); do
        local epoch_dir="$OUTPUT_LOSS_DIR/epoch_$(printf '%04d' $epoch)"
        if [ -f "$epoch_dir/per_sample_losses.npy" ] && [ -f "$epoch_dir/combined_loss_data.csv" ]; then
            ((completed_epochs++))
        fi
    done
    
    echo "Progress: $completed_epochs/$total_epochs epochs completed"
    
    # Check for aggregated results
    if [ -f "$OUTPUT_LOSS_DIR/csl_cifar100_run1_noise_0.01.npy" ] && \
       [ -f "$OUTPUT_LOSS_DIR/metadata_complete_pipeline.json" ]; then
        echo "Aggregated results found: ✅"
        
        if [ $completed_epochs -eq $total_epochs ]; then
            echo "All individual epoch files and aggregated results completed!"
            return 0
        else
            echo "Aggregated results exist but some epoch files missing. Regenerating..."
            return 1
        fi
    else
        echo "Aggregated results missing."
        return 1
    fi
}

# Function to cleanup incomplete results
cleanup_incomplete_results() {
    echo "=== Cleaning up incomplete results ==="
    
    # Remove incomplete epoch directories
    find "$OUTPUT_LOSS_DIR" -type d -name "epoch_*" -exec sh -c '
        epoch_dir="$1"
        if [ ! -f "$epoch_dir/per_sample_losses.npy" ] || [ ! -f "$epoch_dir/combined_loss_data.csv" ]; then
            echo "Removing incomplete epoch directory: $epoch_dir"
            rm -rf "$epoch_dir"
        fi
    ' _ {} \;
    
    # Remove incomplete aggregated files if any epoch files are missing
    local incomplete_found=false
    for epoch in $(seq $START_EPOCH $END_EPOCH); do
        local epoch_dir="$OUTPUT_LOSS_DIR/epoch_$(printf '%04d' $epoch)"
        if [ ! -f "$epoch_dir/per_sample_losses.npy" ]; then
            incomplete_found=true
            break
        fi
    done
    
    if [ "$incomplete_found" = true ]; then
        echo "Removing incomplete aggregated results..."
        rm -f "$OUTPUT_LOSS_DIR/csl_cifar100_run1_noise_0.01.npy"
        rm -f "$OUTPUT_LOSS_DIR/lt_cifar100_run1_noise_0.01.npy" 
        rm -f "$OUTPUT_LOSS_DIR/csl_active_samples.csv"
        rm -f "$OUTPUT_LOSS_DIR/metadata_complete_pipeline.json"
    fi
}

# Main execution - process all 4000 epochs
START_EPOCH=1
END_EPOCH=4000

# Check if generation is already completed
if check_generation_completed $START_EPOCH $END_EPOCH; then
    echo "Loss/Grad generation already completed successfully!"
    echo "If you want to re-run, please delete the existing results first."
else
    # Clean up any incomplete results
    cleanup_incomplete_results
    
    # Run the loss/grad generation
    if run_loss_generation $START_EPOCH $END_EPOCH; then
        echo "=== Loss/Grad Generation Completed Successfully ==="
        
        # Verify all results exist
        echo "=== Verifying Generated Results ==="
        
        # Count generated epoch directories
        EPOCH_DIRS=$(find "$OUTPUT_LOSS_DIR" -type d -name "epoch_*" | wc -l)
        echo "Generated epoch directories: $EPOCH_DIRS"
        
        # Check for required files
        FILES_TO_CHECK=(
            "csl_cifar100_run1_noise_0.01.npy"
            "lt_cifar100_run1_noise_0.01.npy"
            "csl_active_samples.csv"
            "metadata_complete_pipeline.json"
        )
        
        echo "Checking aggregated result files:"
        ALL_FILES_EXIST=true
        for file in "${FILES_TO_CHECK[@]}"; do
            if [ -f "$OUTPUT_LOSS_DIR/$file" ]; then
                echo "  ✅ $file"
            else
                echo "  ❌ $file (MISSING)"
                ALL_FILES_EXIST=false
            fi
        done
        
        if [ "$ALL_FILES_EXIST" = true ]; then
            echo "✅ All required files generated successfully!"
        else
            echo "⚠️  Some files are missing - job may have completed partially"
        fi
        
        # Generate comprehensive summary report
        echo "=== Generating Summary Report ==="
        # Use the conda environment's Python explicitly
        python -c "
import os
import json
import numpy as np
from pathlib import Path

output_dir = '$OUTPUT_LOSS_DIR'
metadata_file = os.path.join(output_dir, 'metadata_complete_pipeline.json')

print(f'Complete Loss/Grad Generation Summary Report:')
print(f'=' * 60)

if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f'Generation Details:')
    print(f'  Model directory: {metadata.get(\"model_dir\", \"N/A\")}')
    print(f'  Total samples: {metadata.get(\"total_samples\", \"N/A\"):,}')
    print(f'  Valid samples: {metadata.get(\"valid_samples\", \"N/A\"):,}')
    print(f'  Processed epochs: {metadata.get(\"num_processed_epochs\", \"N/A\")}')
    print(f'  Processing time: {metadata.get(\"processing_time_seconds\", 0)/3600:.2f} hours')
    print(f'  GPUs used: {metadata.get(\"world_size\", \"N/A\")}')
    print(f'  Class-introduction-aware: {metadata.get(\"class_introduction_aware\", \"N/A\")}')
    print(f'  Sample ID traceable: {metadata.get(\"sample_id_traceable\", \"N/A\")}')
    print(f'  Individual epoch files: {metadata.get(\"individual_epoch_files\", \"N/A\")}')

# Count epoch directories
epoch_dirs = list(Path(output_dir).glob('epoch_*'))
print(f'\\nGenerated Files:')
print(f'  Individual epoch directories: {len(epoch_dirs)}')

# Calculate storage usage
storage_gb = 0
file_counts = {'npy': 0, 'csv': 0, 'json': 0}

for ext in file_counts.keys():
    files = list(Path(output_dir).rglob(f'*.{ext}'))
    file_counts[ext] = len(files)
    storage_gb += sum(f.stat().st_size for f in files) / (1024**3)

print(f'  .npy files: {file_counts[\"npy\"]}')
print(f'  .csv files: {file_counts[\"csv\"]}') 
print(f'  .json files: {file_counts[\"json\"]}')
print(f'  Total storage used: {storage_gb:.2f} GB')

# Load and show CSL statistics if available
csl_file = os.path.join(output_dir, 'csl_cifar100_run1_noise_0.01.npy')
if os.path.exists(csl_file):
    csl = np.load(csl_file)
    valid_csl = csl[csl > 0]  # Only non-zero CSL values
    print(f'\\nAggregated CSL Statistics (valid samples only):')
    print(f'  Valid samples with CSL > 0: {len(valid_csl):,}')
    print(f'  CSL Mean: {valid_csl.mean():.6f}')
    print(f'  CSL Std: {valid_csl.std():.6f}')
    print(f'  CSL Min: {valid_csl.min():.6f}')
    print(f'  CSL Max: {valid_csl.max():.6f}')

# Sample a few epoch directories to show structure
sample_epochs = sorted(epoch_dirs)[:3]
if sample_epochs:
    print(f'\\nSample Epoch Directory Structure:')
    for epoch_dir in sample_epochs:
        print(f'  {epoch_dir.name}/')
        for file in sorted(epoch_dir.glob('*')):
            print(f'    - {file.name}')

print(f'\\n✅ Generation completed successfully!')
print(f'📁 Results location: {output_dir}')
"
        
        echo ""
        echo "=== Files Generated ==="
        echo "Individual epoch files (per epoch):"
        echo "  - epoch_XXXX/per_sample_ids.npy & .csv"
        echo "  - epoch_XXXX/per_sample_losses.npy & .csv" 
        echo "  - epoch_XXXX/per_sample_grads.npy & .csv"
        echo "  - epoch_XXXX/per_sample_labels.npy & .csv"
        echo "  - epoch_XXXX/combined_loss_data.csv"
        echo "  - epoch_XXXX/metadata.json"
        echo ""
        echo "Aggregated results:"
        echo "  - csl_cifar100_run1_noise_0.01.npy (CSL values for all samples)"
        echo "  - lt_cifar100_run1_noise_0.01.npy (Learning time values)"
        echo "  - csl_active_samples.csv (active samples with sample IDs)"
        echo "  - metadata_complete_pipeline.json (complete metadata)"
        
    else
        echo "=== Loss/Grad Generation Failed ==="
        echo "Check error logs for details."
        exit 1
    fi
fi

echo "=== Job Finished ==="
echo "End time: $(date)"
echo "Check results in: $OUTPUT_LOSS_DIR"