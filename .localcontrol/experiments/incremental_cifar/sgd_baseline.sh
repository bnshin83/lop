EXP_NAME="incr_cifar_sgd"
EXP_TYPE="custom"
JOB_NAME="incr_cifar_sgd"
TIME_OVERRIDE="7-00:00:00"

WANDB_PROJECT="upgd-incremental-cifar"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    cd ${PROJECT_DIR}/lop/incremental_cifar
    python incremental_cifar_experiment.py \
        --config ./cfg/base_deep_learning_system.json \
        --verbose --experiment-index 0 \
        --wandb --wandb-project "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-run-name "${WANDB_RUN_NAME}"
}
