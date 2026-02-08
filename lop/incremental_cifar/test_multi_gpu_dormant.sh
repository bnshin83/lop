#!/bin/bash
#SBATCH --job-name=test_dormant
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=0-01:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_test_dormant.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_test_dormant.err

echo "=== Test Multi-GPU Per-Sample Dormant Analysis Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Change to the submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load CUDA module
module load cuda

# Set environment variables for single-node multi-GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29504
export WORLD_SIZE=8
export NCCL_DEBUG=WARN
export NCCL_AVOID_RECORD_STREAMS=0

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Testing New Epoch-per-GPU Approach ==="
echo "Processing epochs 1-8 (1 epoch per GPU)"

# Run the test analysis
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    multi_gpu_dormant_analysis.py \
    --start_epoch 1 \
    --end_epoch 8 \
    --data_path data/cifar-100-python \
    --model_dir results2/base_deep_learning_system/model_parameters \
    --save_base_dir results2/base_deep_learning_system/per_sample_dormant_analysis_test \
    --class_order_file results2/base_deep_learning_system/class_order/index-0.npy \
    --experiment_index 0 \
    --dormant_unit_threshold 0.01

echo "=== Test completed at $(date) ==="