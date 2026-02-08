#!/bin/bash
#SBATCH --job-name=incremental_cifar_git
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=00:01:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_incremental_cifar.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_incremental_cifar.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda python

# Display GPU information
nvidia-smi

# Set environment variable for output directory
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Create results directory for checkpoints
mkdir -p /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results

# Activate Python 3.11 venv (avoids conda Python 3.8 library conflicts)
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate
export PYTHONPATH="/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH"

# Run the Python script with specified arguments
python incremental_cifar_experiment.py --config ./cfg/base_deep_learning_system.json --verbose --experiment-index 0