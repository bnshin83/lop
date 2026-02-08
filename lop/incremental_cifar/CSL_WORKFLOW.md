## Cumulative Sample Loss (CSL) Workflow

This guide explains how CSL is produced in this codebase and how the scripts fit together.
It mirrors the logic used in the notebook `csl-mem/learning_time_tau_v_mem_csl copy.ipynb` while
integrating with this repo.

### What is generated
- **Per-epoch, per-sample loss/grad**: saved as `.npy` files for each training epoch
  (by `score_cifar100_loss.py`).
- **CSL per sample**: the sum of per-sample losses across epochs (by `compute_csl.py`).
- **Learning-time per sample**: a tau-based statistic derived from the running-average of
  per-sample loss-gradients (by `compute_csl.py`, replicating the notebook method). When a
  grad file is missing for a given epoch, the loss values are used as a fallback (same behavior
  as in the notebook).

---

### Components

#### 1) Per-sample loss/grad extraction (pretrained ckpts): `score_cifar100_loss.py`
- Location: `loss-of-plasticity/lop/incremental_cifar/score_cifar100_loss.py`
- Purpose: For each epoch checkpoint, computes per-sample cross-entropy loss and a per-sample
  input-gradient norm. Outputs two `.npy` files per epoch: `loss_*.npy` and `loss_grad_*.npy`.
- Inputs:
  - CIFAR-100 data at `--data_dir` (e.g. `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data`).
  - A directory of checkpoints at `--checkpoint_dir`.
- Outputs:
  - Default output directory: `--output_dir` (default `./results/per_sample_losses/`).
  - Filenames: `loss_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise}.npy` and
    `loss_grad_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise}.npy`.

Example (edit paths as needed):

```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/score_cifar100_loss.py \
  --run_idx 1 \
  --noise_level 0.01 \
  --start_epoch 0 --end_epoch 199 --epoch_step 1 \
  --data_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data \
  --checkpoint_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/pretrained/cifar100 \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/per_sample_losses
```

Notes:
- The script expects a ResNet-18 (CIFAR-100) architecture and a specific checkpoint naming pattern.
- If you want to score the checkpoints from `results/base_deep_learning_system/model_parameters/` (files like
  `index-0_epoch-*.pt`), you must load them with the same architecture they were trained with
  (see `lop/nets/torchvision_modified_resnet.py`). A dedicated scorer for those checkpoints is
  straightforward to add, but not included here by default.

#### 2) Per-sample loss/grad extraction (incremental ckpts): `score_incremental_cifar_loss.py`
- Location: `loss-of-plasticity/lop/incremental_cifar/score_incremental_cifar_loss.py`
- Purpose: For the incremental setup in this repo, load checkpoints from
  `results/<experiment_name>/model_parameters/index-<exp>_epoch-<epoch>.pt`, using the exact architecture
  `lop/nets/torchvision_modified_resnet.build_resnet18`, and compute per-sample loss and input-grad norm
  on the CIFAR-100 training set with the project’s normalization. Writes files per epoch:
  - `loss_cifar100_epoch_{epoch}.npy`
  - `loss_grad_cifar100_epoch_{epoch}.npy`

Example (edits paths as needed):

```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/score_incremental_cifar_loss.py \
  --results_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system \
  --data_path   /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data \
  --experiment_index 0 \
  --start_epoch 0 --end_epoch 4000 \
  --batch_size 256 --num_workers 4 --device auto \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/per_sample_losses_inc \
  --compute_grad true
```

#### 3) CSL and learning-time aggregation: `compute_csl.py`
- Location: `loss-of-plasticity/lop/incremental_cifar/compute_csl.py`
- Purpose: Aggregates the per-epoch `.npy` per-sample losses/gradients to compute:
  - CSL per sample (sum of losses across epochs)
  - Learning-time per sample (tau-based), replicating the notebook’s methodology:
    1) Compute running-average of per-sample loss-gradient over epochs.
    2) Define `tau` as the global mean of that running-average.
    3) Learning-time `lt` is `1 - mean(running_avg < tau)` over epochs, per sample.
- Input directory supports two naming schemes out-of-the-box:
  - Gautschi: `loss_{dataset}_epoch_{epoch}.npy`, `loss_grad_{dataset}_epoch_{epoch}.npy`
  - Gilbreth: `loss_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise}.npy` (and `loss_grad_…`)
  Missing grad files for a given epoch are replaced with the corresponding loss arrays.
- Outputs:
  - `csl_{dataset}_run{run_idx}_noise_{noise}.npy`
  - `lt_{dataset}_run{run_idx}_noise_{noise}.npy` (if grads or fallbacks exist)
  - `metrics_{dataset}_run{run_idx}_noise_{noise}.json` (metadata: `tau`, epochs used, shapes, paths)

Example using per-sample files generated by either scorer:

```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/compute_csl.py \
  --input_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/per_sample_losses \
  --dataset cifar100 --run_idx 1 --noise_level 0.01 \
  --start_epoch 0 --end_epoch 199 --epoch_step 1 \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/csl
```

If you already have per-sample `.npy` files from the `csl-mem` project (e.g., under
`/scratch/gautschi/shin283/csl-mem/results/per_sample_losses`), point `--input_dir` there instead.

---

### Unified one-shot pipeline: `csl_pipeline.py`
- Location: `loss-of-plasticity/lop/incremental_cifar/csl_pipeline.py`
- Purpose: Single command to either:
  - score per-sample losses from incremental `model_parameters` and then aggregate; or
  - use an existing directory of per-sample files and aggregate.

Examples:

- From model_parameters (scores a range of epochs, then aggregates):
```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/csl_pipeline.py \
  --source model_parameters \
  --results_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system \
  --data_path   /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data \
  --experiment_index 0 \
  --start_epoch 0 --end_epoch 4000 --epoch_step 200 \
  --batch_size 256 --num_workers 4 --device auto --compute_grad true \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/csl
```

- From an existing per-sample directory:
```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/csl_pipeline.py \
  --source per_samples \
  --input_dir /scratch/gautschi/shin283/csl-mem/results/per_sample_losses \
  --dataset cifar100 --run_idx 1 --noise_level 0.01 \
  --start_epoch 0 --end_epoch 199 --epoch_step 1 \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/csl
```

---

### End-to-end workflow

1) Ensure CIFAR-100 data is present
   - Directory: `/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data`

2) Generate per-sample loss/grad per epoch
   - Option A (pretrained ckpts): run `score_cifar100_loss.py`.
   - Option B (incremental ckpts): run `score_incremental_cifar_loss.py`.
   - Option C: use `csl_pipeline.py --source model_parameters` for one-shot scoring + aggregation.

3) Compute CSL and learning-time
   - Run `compute_csl.py` on the directory containing the per-sample `.npy` files, or use `csl_pipeline.py`.
   - Outputs written to `results/csl/` include `csl_*.npy`, `lt_*.npy`, and a metadata JSON with `tau`.

4) (Optional) Plot/analyze
   - See notebook `csl-mem/learning_time_tau_v_mem_csl copy.ipynb` for plotting examples (e.g., histograms of
     learning-time, binned analyses against memory scores, etc.).

---

### Tips and caveats
- For exact architectural parity with the incremental setup, prefer `score_incremental_cifar_loss.py` (or the
  unified `csl_pipeline.py --source model_parameters`). These use `build_resnet18` from
  `lop/nets/torchvision_modified_resnet.py` and the project’s CIFAR normalization.
- `compute_csl.py` will proceed even if some grad files are missing (per-epoch fallback to loss values), exactly
  as done in the notebook.

---

### Quick test (two checkpoints)

To sanity-check on two epochs only (e.g., 0 and 200):

```bash
python3 /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/csl_pipeline.py \
  --source model_parameters \
  --results_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system \
  --data_path   /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/data \
  --experiment_index 0 \
  --start_epoch 0 --end_epoch 200 \
  --batch_size 256 --num_workers 4 --device auto --compute_grad false \
  --output_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/csl_test_small
```

This will:
- Score per-sample loss for available epochs in [0, 200] (e.g., 0 and 200),
- Aggregate to CSL (and learning-time if grads are enabled),
- Save outputs and a metadata JSON under `results/csl_test_small`.


