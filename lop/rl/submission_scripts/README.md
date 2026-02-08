# submission_scripts

This directory contains helper scripts used for running the reinforcement learning experiments in `lop/rl`. Each script focuses on plotting metrics or launching training runs with different environments/agents.

## Contents
- `plot_all_metrics_v2.py` – Aggregates and plots the recorded metrics for every training run.
- `plot_dormant_correct.py` – Plots the "dormant correct" metric for diagnosis and comparison.
- `plot_returns.py` – Generates return curves for a single experiment trace.
- `run_ant.sh` – Shell wrapper to launch the Ant training job.
- `run_ppo_wandb.py` – Training script that integrates PPO agent runs with a Weights & Biases logger.
- `run_sant.sh` – Shell wrapper to launch the SAnt training job.

Run these scripts from this folder so relative paths resolve correctly. Review each script for usage notes before executing them.