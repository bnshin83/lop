#!/bin/bash
#SBATCH --job-name=batch_run_curvature
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=14     
#SBATCH --gpus-per-node=4
#SBATCH --time=13-1:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_batch_run_curvature.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_batch_run_curvature.err


# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load cuda

# Display GPU information
nvidia-smi

# Set environment variables
export OUTPUT_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Set OpenMP threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the post-analysis script after training completes
python3.8 post_run_analysis_modified2.py --results_dir ./results/continual_backpropagation/