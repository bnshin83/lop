#!/bin/bash
#SBATCH --job-name=multi_gpu_csl
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_csl.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_multi_gpu_csl.err

echo "=== Multi-GPU CSL Analysis Job Started ==="
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
export MASTER_PORT=29501
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
INPUT_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_inc"
DATA_PATH="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data"
OUTPUT_CSL_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/multi_gpu_csl_results"
CLASS_ORDER_FILE="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.npy"

echo "Input directory (per_sample_losses_inc): $INPUT_DIR"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_CSL_DIR"
echo "Class order file: $CLASS_ORDER_FILE"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
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
mkdir -p "$OUTPUT_CSL_DIR"
echo "Created output directory: $OUTPUT_CSL_DIR"

# Count available loss files
LOSS_COUNT=$(find "$INPUT_DIR" -name "loss_cifar100_epoch_*.npy" | wc -l)
echo "Found $LOSS_COUNT loss files in $INPUT_DIR"

if [ $LOSS_COUNT -eq 0 ]; then
    echo "ERROR: No loss files found in $INPUT_DIR"
    exit 1
fi

echo "========================"

# Function to run the CSL analysis with error handling
run_csl_analysis() {
    local start_epoch=$1
    local end_epoch=$2
    
    echo "=== Starting CSL Analysis ==="
    echo "Epoch range: $start_epoch to $end_epoch"
    echo "Start time: $(date)"
    
    # Run the multi-GPU CSL analysis script using torchrun for proper DDP setup
    torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        multi_gpu_csl_analysis.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_CSL_DIR" \
        --data_path "$DATA_PATH" \
        --class_order_file "$CLASS_ORDER_FILE" \
        --experiment_index 0 \
        --start_epoch $start_epoch \
        --end_epoch $end_epoch \
        --dataset cifar100
    
    local exit_code=$?
    echo "CSL Analysis completed with exit code: $exit_code"
    echo "End time: $(date)"
    
    return $exit_code
}

# Function to check if CSL results already exist
check_csl_completed() {
    if [ -f "$OUTPUT_CSL_DIR/csl_results_full.npy" ] && [ -f "$OUTPUT_CSL_DIR/metadata_cifar100_csl.json" ]; then
        echo "=== CSL Results Already Exist ==="
        echo "Full results file: $OUTPUT_CSL_DIR/csl_results_full.npy"
        echo "Metadata file: $OUTPUT_CSL_DIR/metadata_cifar100_csl.json"
        return 0
    else
        return 1
    fi
}

# Main execution - process all available epochs
START_EPOCH=1
END_EPOCH=2

# Check if results already exist
if check_csl_completed; then
    echo "CSL analysis already completed successfully!"
    echo "If you want to re-run, please delete the existing results first."
else
    # Run the CSL analysis
    if run_csl_analysis $START_EPOCH $END_EPOCH; then
        echo "=== CSL Analysis Completed Successfully ==="
        
        # Generate summary report
        echo "=== Generating Summary Report ==="
        $CONDA_PREFIX/bin/python -c "
import os
import json
import numpy as np
from pathlib import Path

output_dir = '$OUTPUT_CSL_DIR'
metadata_file = os.path.join(output_dir, 'metadata_cifar100_csl.json')

if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f'CSL Analysis Summary Report:')
    print(f'  Dataset: {metadata.get(\"dataset\", \"N/A\")}')
    print(f'  Total samples: {metadata.get(\"total_samples\", \"N/A\"):,}')
    print(f'  Valid samples: {metadata.get(\"valid_samples\", \"N/A\"):,}')
    print(f'  Processed epochs: {metadata.get(\"num_processed_epochs\", \"N/A\")}')
    print(f'  Processing time: {metadata.get(\"processing_time_seconds\", 0)/3600:.2f} hours')
    print(f'  World size (GPUs used): {metadata.get(\"world_size\", \"N/A\")}')
    print(f'  Class-introduction-aware: {metadata.get(\"class_introduction_aware\", \"N/A\")}')
    print(f'  Sample ID traceable: {metadata.get(\"sample_id_traceable\", \"N/A\")}')
    
    # Calculate storage used
    storage_gb = sum(f.stat().st_size for f in Path(output_dir).rglob('*.npy')) / (1024**3)
    storage_gb += sum(f.stat().st_size for f in Path(output_dir).rglob('*.csv')) / (1024**3)
    storage_gb += sum(f.stat().st_size for f in Path(output_dir).rglob('*.json')) / (1024**3)
    print(f'  Storage used: {storage_gb:.2f} GB')
    
    # Load and show CSL statistics
    csl_file = os.path.join(output_dir, 'csl_cifar100_run1_noise_0.01.npy')
    if os.path.exists(csl_file):
        csl = np.load(csl_file)
        valid_csl = csl[csl > 0]  # Only non-zero CSL values
        print(f'  CSL Statistics (valid samples only):')
        print(f'    Mean: {valid_csl.mean():.4f}')
        print(f'    Std: {valid_csl.std():.4f}')
        print(f'    Min: {valid_csl.min():.4f}')
        print(f'    Max: {valid_csl.max():.4f}')
else:
    print('ERROR: Metadata file not found!')
"
        
        echo ""
        echo "=== Files Generated ==="
        echo "Main results:"
        echo "  - csl_results_full.npy (complete results with metadata)"
        echo "  - csl_cifar100_run1_noise_0.01.npy (CSL values for all samples)"
        echo "  - lt_cifar100_run1_noise_0.01.npy (Learning time values)"
        echo "  - csl_active_samples.csv (active samples only with sample IDs)"
        echo "  - metadata_cifar100_csl.json (processing metadata)"
        
    else
        echo "=== CSL Analysis Failed ==="
        echo "Check error logs for details."
        exit 1
    fi
fi

echo "=== Job Finished ==="
echo "End time: $(date)"
echo "Check results in: $OUTPUT_CSL_DIR"