#!/bin/bash
#SBATCH --job-name=csl_lt_progression
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14      # Balanced for GPU workload
#SBATCH --gpus-per-node=2       # Use 2 GPUs for better performance
#SBATCH --time=02:00:00         # Reduced time with GPU acceleration
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_progression.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_progression.err


# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load cuda

# Display GPU information
nvidia-smi
echo "Number of GPUs available: $SLURM_GPUS_PER_NODE"

# Set environment variables
export OUTPUT_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop
export PYTHONPATH=/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Set OpenMP threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Print memory and CPU info
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: Auto-allocated"

# Print Python and conda environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Run the progression computation with GPU acceleration
echo "Starting CSL/LT progression computation with GPU acceleration..."
python3 compute_progression_data.py \
    --per_sample_dir $OUTPUT_DIR/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_inc \
    --output_dir $OUTPUT_DIR/incremental_cifar/results2/progression_cache \
    --dataset cifar100 \
    --device auto \
    --gpu_memory_fraction 0.8

# Check if computation was successful
if [ $? -eq 0 ]; then
    echo "✅ Progression computation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/incremental_cifar/results2/progression_cache/"
    ls -la $OUTPUT_DIR/incremental_cifar/results2/progression_cache/
else
    echo "❌ Progression computation failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed."
