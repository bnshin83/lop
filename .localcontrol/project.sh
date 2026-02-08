PROJECT_NAME="loss-of-plasticity"
# Remote dir: ${SCRATCH}/${PROJECT_NAME}
# i.e. /scratch/gautschi/shin283/loss-of-plasticity

# Exclude when pushing code TO cluster
RSYNC_EXCLUDE=(
    ".git" "__pycache__" "*.pyc" "*.pyo"
    "*.pt" "*.pth" "*.ckpt" "*.npy"
    "wandb" "data" ".DS_Store"
    "results" ".lop_venv_compute"
    "lop.egg-info"
)

# Pull config: what to pull FROM cluster
PULL_REMOTE_DIR="lop/incremental_cifar/results"
PULL_LOCAL_DIR="results"
PULL_INCLUDE=("*.json" "*.csv" "*.out" "*.err" "*.txt" "*.log" "*.npy")
PULL_EXCLUDE=("*.pt" "*.pth" "*.ckpt")

# Archive destinations
ARCHIVE_DEPOT="/depot/jhaddock/data/shin283/lop"
ARCHIVE_NAS=""

# Override cluster env_setup — lop uses its own venv
env_setup() {
    cat <<ENVEOF
module load cuda python
source ${SCRATCH}/${PROJECT_NAME}/.lop_venv_compute/bin/activate
ENVEOF
}
