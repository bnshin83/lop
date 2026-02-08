EXP_NAME="incr_cifar_gating"
EXP_TYPE="custom"
JOB_NAME="incr_cifar_upgd_gating"
TIME_OVERRIDE="7-00:00:00"
CUSTOM_ARRAY="0-8"

WANDB_PROJECT="upgd-incremental-cifar"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    declare -A gating_modes=([0]="full" [1]="full" [2]="full"
        [3]="output_only" [4]="output_only" [5]="output_only"
        [6]="hidden_only" [7]="hidden_only" [8]="hidden_only")
    declare -A seeds=([0]=0 [1]=1 [2]=2 [3]=0 [4]=1 [5]=2 [6]=0 [7]=1 [8]=2)

    GATING_MODE=${gating_modes[$SLURM_ARRAY_TASK_ID]}
    SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

    CONFIG_FILE="/tmp/upgd_gating_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
    cat > $CONFIG_FILE <<CFGEOF
{
  "experiment_name": "upgd_gating_${GATING_MODE}",
  "num_workers": 12, "stepsize": 0.1, "weight_decay": 0.0005,
  "momentum": 0.9, "noise_std": 0.0,
  "use_upgd": true, "upgd_beta_utility": 0.999, "upgd_sigma": 0.001,
  "upgd_beta1": 0.9, "upgd_beta2": 0.999, "upgd_eps": 1e-5,
  "upgd_use_adam_moments": true, "upgd_gating_mode": "${GATING_MODE}",
  "upgd_non_gated_scale": 0.5,
  "use_cbp": false, "reset_head": false, "reset_network": false,
  "early_stopping": true
}
CFGEOF

    cd ${PROJECT_DIR}/lop/incremental_cifar
    python incremental_cifar_experiment.py \
        --config $CONFIG_FILE \
        --verbose --experiment-index $SEED \
        --wandb --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-run-name "${WANDB_RUN_NAME}"
    rm -f $CONFIG_FILE
}
