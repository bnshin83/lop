#!/bin/bash
#SBATCH --job-name=post_run_analysis
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28      # Increased for better data loading
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_post_run_analysis.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_post_run_analysis.err


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
python post_run_analysis_modified2.py \
    --results_dir $OUTPUT_DIR/incremental_cifar/results/continual_backpropagation