#!/bin/bash
#SBATCH --job-name=generate_evolution_plots
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1       # Use 2 GPUs for better performance
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_plots.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_plots.err

# Job to generate all CSL and Learning-Time evolution plots efficiently
# This script runs the Python plotting script in a batch environment

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

# Run the plotting script with enhanced visualization parameters
echo "Starting CSL/LT evolution plot generation with enhanced techniques..."

# Generate multiple visualization styles for comprehensive analysis
echo "📊 Generating density heatmap visualization..."
python3 generate_evolution_plots.py \
    --results_dir $OUTPUT_DIR/incremental_cifar/results2 \
    --output_dir $OUTPUT_DIR/incremental_cifar/evolution_plots/density \
    --csl_plot_style density \
    --subsample_individual 20000 \
    --subsample_summary 3000 \
    --csl_temporal_downsample 10 \
    --csl_cmap hot

echo "📊 Generating ridge plot visualization..."
python3 generate_evolution_plots.py \
    --results_dir $OUTPUT_DIR/incremental_cifar/results2 \
    --output_dir $OUTPUT_DIR/incremental_cifar/evolution_plots/ridges \
    --csl_plot_style ridges \
    --subsample_individual 15000 \
    --subsample_summary 3000 \
    --csl_temporal_downsample 20

echo "📊 Generating volatility-colored lines..."
python3 generate_evolution_plots.py \
    --results_dir $OUTPUT_DIR/incremental_cifar/results2 \
    --output_dir $OUTPUT_DIR/incremental_cifar/evolution_plots/volatility \
    --csl_plot_style lines \
    --csl_color_mode volatility \
    --csl_highlight_k 25 \
    --subsample_individual 5000 \
    --subsample_summary 3000 \
    --csl_temporal_downsample 5 \
    --csl_density_mode \
    --csl_cmap plasma

echo "📊 Generating fast percentile bands..."
python3 generate_evolution_plots.py \
    --results_dir $OUTPUT_DIR/incremental_cifar/results2 \
    --output_dir $OUTPUT_DIR/incremental_cifar/evolution_plots/percentiles \
    --csl_plot_style percentiles \
    --csl_percentiles_subsample 25000 \
    --subsample_summary 3000 \
    --csl_temporal_downsample 10

# Check if plot generation was successful
if [ $? -eq 0 ]; then
    echo "✅ All plot generation completed successfully!"
    echo "Results saved to multiple directories under: $OUTPUT_DIR/incremental_cifar/evolution_plots/"
    echo ""
    echo "📁 Generated visualizations:"
    echo "   - Density heatmaps: evolution_plots/density/"
    echo "   - Ridge plots: evolution_plots/ridges/"
    echo "   - Volatility analysis: evolution_plots/volatility/"
    echo "   - Percentile bands: evolution_plots/percentiles/"
    echo ""
    echo "Directory contents:"
    find $OUTPUT_DIR/incremental_cifar/evolution_plots/ -name "*.png" -exec ls -lh {} \; 2>/dev/null || echo "Plot files not found"
else
    echo "❌ Plot generation failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed."