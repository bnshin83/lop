#!/bin/bash
#SBATCH --job-name=ant_std_s0
#SBATCH --account=jhaddock
#SBATCH --partition=ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=3-0:00:00
#SBATCH --output=/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/%j_ant_std.out
#SBATCH --error=/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/%j_ant_std.err


CONFIG="std"

SEED=0

set -e

PROJECT_DIR="/scratch/gautschi/shin283/loss-of-plasticity"
RL_DIR="${PROJECT_DIR}/lop/rl"
LOG_DIR="${RL_DIR}/logs"

mkdir -p ${LOG_DIR}
cd ${RL_DIR}

module load cuda python

# MuJoCo setup
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

# Activate Python 3.11 venv (avoids conda Python 3.8 library conflicts)
source /scratch/gautschi/shin283/loss-of-plasticity/.lop_venv_compute/bin/activate
export PYTHONPATH=${PROJECT_DIR}:$PYTHONPATH

export WANDB_ENTITY="minds_rl"
export WANDB_PROJECT="loss-of-plasticity-rl"
export WANDB_API_KEY=""
export WANDB_MODE="online"
export WANDB_RUN_NAME="ant_${CONFIG}_s${SEED}"

echo "========================================="
echo "Ant-v3 | ${CONFIG} | seed ${SEED}"
echo "Job: ${SLURM_JOB_ID} | $(date)"
echo "WandB: ${WANDB_RUN_NAME}"
echo "========================================="

python run_ppo_wandb.py \
    --config cfg/ant/${CONFIG}.yml \
    --seed ${SEED} \
    --device cuda

echo "Done | $(date)"
