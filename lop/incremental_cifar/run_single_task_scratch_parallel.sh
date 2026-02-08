#!/bin/bash
#SBATCH --job-name=single_task_scratch
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
# #SBATCH --array=0-3%2
# Array mode disabled for single task on single GPU
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_single_task_scratch.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/log/%j_single_task_scratch.err

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load CUDA and Python modules
module load cuda
module load python

# Display GPU information
nvidia-smi

# Set environment variable for output directory
export OUTPUT_DIR=/scratch/gautschi/shin283/lop

# Configuration - ENABLE memo ordering to debug the issue
NO_ORDERING="true"  # Enable memo ordering to debug
MEMO_ORDER="low_to_high"  # Sample memorization score ordering strategy
SCRATCH_MODE="true"
EPOCHS_PER_TASK="1000"  # Set to desired epochs for single task
INCREMENTAL_EPOCHS="false"

# Single task mode configuration
SINGLE_TASK_MODE="true"
SINGLE_TASK_NUMBER=19  # Which task to run (0-19)

# Set task ID for single task mode (after SINGLE_TASK_NUMBER is defined)
TASK_ID=$SINGLE_TASK_NUMBER

# Note: Class order is handled by memo ordering when NO_ORDERING=false

# Create results directory based on experiment mode
BASE_DIR="/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar"

# Build directory name components
if [ "$NO_ORDERING" = "true" ]; then
    DIR_NAME="results_single_task_no_ordering"
else
    DIR_NAME="results_single_task_memo_ordered_${MEMO_ORDER}"
fi

# Add scratch mode to directory name
if [ "$SCRATCH_MODE" = "true" ]; then
    DIR_NAME="${DIR_NAME}_scratch"
fi

# Add epochs mode to directory name
if [ "$INCREMENTAL_EPOCHS" = "true" ]; then
    DIR_NAME="${DIR_NAME}_incremental_${EPOCHS_PER_TASK}"
else
    DIR_NAME="${DIR_NAME}_fixed_${EPOCHS_PER_TASK}"
fi

# Add single task identifier if in single task mode
if [ "$SINGLE_TASK_MODE" = "true" ]; then
    DIR_NAME="${DIR_NAME}_single_task_${SINGLE_TASK_NUMBER}"
fi

RESULTS_DIR="${BASE_DIR}/${DIR_NAME}"

mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/task_${TASK_ID}

# Activate conda environment
source /scratch/gautschi/shin283/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/gautschi/shin283/conda_envs/lop

echo "========================================================================"
echo "Single Task Scratch Mode - Single GPU"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $TASK_ID"
echo "Config: ./cfg/base_deep_learning_system.json"
echo "Epochs: $EPOCHS_PER_TASK"
echo "Memo Order: $MEMO_ORDER"
echo "No Ordering: $NO_ORDERING"
echo "Results dir: $RESULTS_DIR"
echo "========================================================================"

# Build command arguments
COMMON_ARGS="--config ./cfg/base_deep_learning_system.json --results_dir ${RESULTS_DIR}/task_${TASK_ID} --epochs $EPOCHS_PER_TASK --gpu_id 0"

# Add memo ordering arguments  
COMMON_ARGS="$COMMON_ARGS --memo_order $MEMO_ORDER"

# Add no_ordering if enabled
if [ "$NO_ORDERING" = "true" ]; then
    COMMON_ARGS="$COMMON_ARGS --no_ordering"
fi

# Run the single task experiment
echo "Starting Task $TASK_ID at $(date)"
python3.8 single_task_scratch_experiment.py --task_id $TASK_ID $COMMON_ARGS

# Check if the task completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Task $TASK_ID completed successfully at $(date)"
else
    echo "❌ Task $TASK_ID failed at $(date)"
    exit 1
fi

echo "========================================================================"
echo "Task $TASK_ID finished at $(date)"
echo "========================================================================"