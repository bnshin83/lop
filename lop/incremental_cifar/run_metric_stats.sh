#!/bin/bash
#SBATCH --job-name=metric_stats
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_metric_stats.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_metric_stats.err


# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda

# Display GPU info
nvidia-smi

# Set environment variables
export OUTPUT_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

echo "--- Environment Diagnostics Start ---"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION: $CUDA_VERSION"

echo "which python: $(which python)"
echo "python version: $(python --version)"

echo "conda list pytorch:"
conda list pytorch

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo "which nvcc: $(which nvcc)"
if [ -x "$(which nvcc)" ]; then nvcc --version; fi

echo "Running nvidia-smi again just before python script:"
nvidia-smi

echo "--- Environment Diagnostics End ---"



# Add project root to Python path so 'lop' module can be found
export PYTHONPATH=/scratch/gautschi/shin283/loss-of-plasticity:$PYTHONPATH

# Set environment variables for directories
export RESULTS_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system
export OUTPUT_DIR=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/output

# Make sure the output directories exist
mkdir -p $OUTPUT_DIR/metric_stats/base_deep_learning_system

# Verify class order file exists
ls -la $RESULTS_DIR/class_order/index-0.npy || echo "WARNING: Class order NPY file not found!"

# Run the statistics analyzer with compute phase first
python metric_stats_analyzer.py \
    --results_dir $RESULTS_DIR \
    --output_dir $OUTPUT_DIR/metric_stats/base_deep_learning_system \
    --data_path lop/incremental_cifar/data \
    --compute

# # Generate plots for each combination of metrics and statistics
# declare -a METRICS=("accuracy" "confidence" "true_class_confidence" "effective_rank" "stable_rank" "dormant_units_proportion" "weight_magnitude")
# declare -a STATISTICS=("mean" "max" "median" "std" "q25" "q75" "skewness" "kurtosis")

# for METRIC in "${METRICS[@]}"; do
#     for STAT in "${STATISTICS[@]}"; do
#         echo "Generating plots for $METRIC - $STAT..."
#         python metric_stats_analyzer.py \
#             --results_dir $RESULTS_DIR \
#             --output_dir $OUTPUT_DIR/metric_stats/base_deep_learning_system \
#             --plot_only --metric $METRIC --statistic $STAT \
#             --log_dir lop/incremental_cifar/log
#     done
# done


