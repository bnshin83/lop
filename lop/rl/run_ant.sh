#!/bin/bash
#SBATCH --job-name=ant_upgd_full
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=3-0:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/%j_ant_upgd_full.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/%j_ant_upgd_full.err

# ============================================
# PPO Ant-v3 Experiment
# Usage: sbatch run_ant.sh
# ============================================

# ============================================
# CHOOSE CONFIG (uncomment one)
# ============================================
# CONFIG="std"
# CONFIG="ns"
# CONFIG="l2"
# CONFIG="cbp"
CONFIG="upgd_full"
# CONFIG="upgd_full"

# ============================================
# CHOOSE SEED
# ============================================
SEED=0

# ============================================
set -e

PROJECT_DIR="/scratch/gautschi/shin283/loss-of-plasticity"
RL_DIR="${PROJECT_DIR}/lop/rl"
LOG_DIR="${RL_DIR}/logs"

mkdir -p ${LOG_DIR}
cd ${RL_DIR}

# Environment setup - use compute-node venv
module load cuda
module load python

# Set PYTHONPATH before activating environment
export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH

# Activate the compute-node venv (created on compute node for compatibility)
source ${PROJECT_DIR}/.lop_venv_compute/bin/activate

# MuJoCo setup
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

# WandB setup
export WANDB_ENTITY="minds_rl"
export WANDB_PROJECT="loss-of-plasticity-rl"
export WANDB_API_KEY="9ac056cc70ed02df5b4c069e79ebedf6cf17605d"
export WANDB_MODE="online"
export WANDB_RUN_NAME="ant_${CONFIG}_s${SEED}"

echo "========================================="
echo "Ant-v3 | ${CONFIG} | seed ${SEED}"
echo "Job: ${SLURM_JOB_ID} | $(date)"
echo "WandB: ${WANDB_RUN_NAME}"
echo "========================================="

export PYTHONUNBUFFERED=1

python -u run_ppo_wandb.py \
    --config cfg/ant/${CONFIG}.yml \
    --seed ${SEED} \
    --device cuda \
    --no-resume \
    --run-suffix fresh_001

echo "Done | $(date)"
