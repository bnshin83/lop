#!/bin/bash
#SBATCH --job-name=recalc_test_acc
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_recalc_test_acc.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_recalc_test_acc.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Set memo percentage to recalculate (change this value as needed)
MEMO_PERCENT=90

# Set whether to use all test data for corresponding classes (true/false)
USE_ALL_DATA=false

# Set whether to use multi-GPU distributed processing (true/false)
MULTI_GPU=true

# Set environment variables for single-node multi-GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29504
export WORLD_SIZE=8
export NCCL_DEBUG=WARN
export NCCL_AVOID_RECORD_STREAMS=0

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Build the command with conditional flags
if [ "$MULTI_GPU" = "true" ]; then
    # Multi-GPU command using torchrun
    CMD="torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        recalculate_test_accuracy_with_memo_filter.py \
        --results_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_low_memo_${MEMO_PERCENT}pct \
        --memo_percent ${MEMO_PERCENT} \
        --run_index 0 \
        --multi_gpu"
    
    echo "Running with multi-GPU (8 GPUs) distributed processing"
else
    # Single GPU command
    CMD="python recalculate_test_accuracy_with_memo_filter.py \
        --results_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_low_memo_${MEMO_PERCENT}pct \
        --memo_percent ${MEMO_PERCENT} \
        --run_index 0 \
        --gpu_id 0"
    
    echo "Running with single GPU"
fi

if [ "$USE_ALL_DATA" = "true" ]; then
    CMD="$CMD --use_all_data"
    echo "Using all test samples for corresponding classes"
else
    echo "Using memo filtering (${MEMO_PERCENT}% of test samples)"
fi

echo "Command: $CMD"
echo "Starting test accuracy recalculation..."

# Run the test accuracy recalculation script
eval $CMD