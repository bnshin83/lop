#!/bin/bash
#SBATCH --job-name=csl_pipeline
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28      # Increased for better data loading
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_csl_pipeline.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_csl_pipeline.err


# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load cuda

# Display GPU information
nvidia-smi

# Set environment variables
export OUTPUT_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop
export PYTHONPATH=/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

# Set OpenMP threads for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the CSL pipeline script directly using the checkpoints in results2
python3 csl_pipeline.py \
    --source model_parameters \
    --results_dir $OUTPUT_DIR/incremental_cifar/results2 \
    --data_path $OUTPUT_DIR/incremental_cifar/data \
    --experiment_index 0 \
    --start_epoch 0 \
    --end_epoch 4000 \
    --epoch_step 200 \
    --batch_size 256 \
    --num_workers 4 \
    --device cuda \
    --compute_grad true \
    --output_dir $OUTPUT_DIR/incremental_cifar/results2/csl
