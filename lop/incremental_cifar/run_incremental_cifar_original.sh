#!/bin/bash
#SBATCH --job-name=inc_cifar_orig_idx0_ep200
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=13-1:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_inc_cifar_orig_idx0_ep200.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_inc_cifar_orig_idx0_ep200.err

# Parameters - modify these to change the experiment
EXPERIMENT_INDEX=0
TASK_EPOCHS=200

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Set environment variable for output directory
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Create results directory for original experiment
RESULTS_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_original_idx${EXPERIMENT_INDEX}_ep${TASK_EPOCHS}"
mkdir -p ${RESULTS_DIR}

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Run the original experiment
python3.8 incremental_cifar_experiment_original.py \
    --config ./cfg/base_deep_learning_system.json \
    --experiment-index ${EXPERIMENT_INDEX} \
    --task-epochs ${TASK_EPOCHS} \
    --results-dir ${RESULTS_DIR} \
    --verbose