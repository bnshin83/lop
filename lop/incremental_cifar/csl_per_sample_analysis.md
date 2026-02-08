I looked through your job scripts and the file layout you’re writing. You’ve got everything needed to quantify how per-sample curvature and cumulative sample loss (CSL) relate to plasticity loss at each task boundary (every 200 epochs). Below is a drop-in, streaming analysis that (a) builds compact per-task features from your 50k×4k data, and (b) measures how those features predict the “loss jump” at each boundary, which is a good proxy for plasticity loss.

### What we compute
- **Boundary plasticity loss proxy** (kept from L7): for boundary between task t and t+1,
  - Δloss = loss[e = t·200 + 1] − loss[e = t·200] on previously introduced classes.
- **Per-sample, per-task features (computed over epochs within task t):**
  - Curvature: mean, max, slope (trend within the task)
  - Loss: mean, max, slope
  - Per-sample gradient magnitude: mean, max, slope (auto-detected if `per_sample_grads.npy` exists)
  - CSL-to-boundary: cumulative per-sample loss up to the boundary (prefix sum)
- **Association metric**: Spearman correlation (rank-based) between Δloss and each feature per task.

### Quick note on Spearman
- Spearman correlation is Pearson applied to the ranks of the variables; it captures monotonic relationships and is robust to outliers and nonlinearity.

The code streams epoch files with memory-mapped loads, so it’s fast and memory-safe.

```python
# analyze_plasticity_curvature_csl.py
import os, argparse, json
import numpy as np
from pathlib import Path

def np_load(path):
    return np.load(path, allow_pickle=False, mmap_mode='r')

def find_first_file(base_dir, rel_path):
    for p in sorted(Path(base_dir).glob('epoch_*')):
        f = p / rel_path
        if f.exists():
            return str(f)
    return None

def rankdata_ties_avg(a):
    a = np.asarray(a)
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    diffs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    idx = np.where(diffs)[0]
    counts = np.diff(np.r_[idx, len(a_sorted)])
    starts = idx
    ends = idx + counts
    ranks = np.empty(len(a), dtype=float)
    for s, e in zip(starts, ends):
        ranks[s:e] = (s + e - 1) / 2.0
    return ranks[inv]

def spearmanr_np(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    rx = rankdata_ties_avg(x[m])
    ry = rankdata_ties_avg(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0,1]

def list_epochs(base_dir):
    eps = []
    for p in Path(base_dir).glob('epoch_*'):
        try:
            eps.append(int(p.name.split('_')[-1]))
        except:
            pass
    return sorted(eps)

def load_grad_scalar(path, chunk=4096):
    arr = np_load(path)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    axes = tuple(range(1, arr.ndim))
    for i in range(0, n, chunk):
        j = min(i+chunk, n)
        sl = (slice(i, j),) + tuple(slice(None) for _ in axes)
        a = np.asarray(arr[sl])
        out[i:j] = np.sqrt((a*a).sum(axis=axes))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loss_dir', required=True)
    ap.add_argument('--curv_dir', required=True)
    ap.add_argument('--class_order_file', required=True)
    ap.add_argument('--epochs_per_task', type=int, default=200)
    ap.add_argument('--classes_per_task', type=int, default=5)
    ap.add_argument('--num_tasks', type=int, default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--grad_filename', default='per_sample_grads.npy')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    curv_epochs = list_epochs(args.curv_dir)
    loss_epochs = list_epochs(args.loss_dir)
    if not curv_epochs or not loss_epochs:
        raise RuntimeError('No epoch_* directories found in curv_dir or loss_dir')

    max_epoch = min(curv_epochs[-1], loss_epochs[-1])
    if args.num_tasks is None:
        args.num_tasks = max_epoch // args.epochs_per_task

    labels_path = find_first_file(args.loss_dir, 'per_sample_labels.npy')
    if labels_path is None:
        raise FileNotFoundError('per_sample_labels.npy not found in loss_dir/epoch_*')
    labels = np.load(labels_path)
    n = labels.shape[0]

    class_order = np.load(args.class_order_file)
    inv_order = np.empty(class_order.shape[0], dtype=int)
    inv_order[class_order] = np.arange(class_order.shape[0])
    class_task = inv_order // args.classes_per_task
    sample_intro_task = class_task[labels]

    W = args.epochs_per_task
    Sx = W*(W+1)/2.0
    Sx2 = W*(W+1)*(2*W+1)/6.0
    denomW = (W*Sx2 - Sx*Sx)

    csl_prefix = np.zeros(n, dtype=np.float64)

    rows = []
    qrows = []

    for t in range(args.num_tasks):
        start_e = t*W + 1
        end_e   = (t+1)*W

        sum_curv = np.zeros(n, dtype=np.float64)
        sumxy_curv = np.zeros(n, dtype=np.float64)
        max_curv = np.full(n, -np.inf, dtype=np.float64)

        sum_loss_feat = np.zeros(n, dtype=np.float64)
        sumxy_loss_feat = np.zeros(n, dtype=np.float64)
        max_loss_feat = np.full(n, -np.inf, dtype=np.float64)
        sum_loss_for_csl = np.zeros(n, dtype=np.float64)

        have_grad = True
        sum_grad = np.zeros(n, dtype=np.float64)
        sumxy_grad = np.zeros(n, dtype=np.float64)
        max_grad = np.full(n, -np.inf, dtype=np.float64)

        w_count = 0
        for i, e in enumerate(range(start_e, end_e+1), start=1):
            curv_path = Path(args.curv_dir) / f'epoch_{e:04d}' / 'per_image_curvature.npy'
            loss_path = Path(args.loss_dir) / f'epoch_{e:04d}' / 'per_sample_losses.npy'
            grad_path = Path(args.loss_dir) / f'epoch_{e:04d}' / args.grad_filename

            if not curv_path.exists() or not loss_path.exists():
                continue

            curv = np_load(str(curv_path))
            loss = np_load(str(loss_path))

            sum_curv += curv
            sumxy_curv += i * curv
            np.maximum(max_curv, curv, out=max_curv)

            sum_loss_feat += loss
            sumxy_loss_feat += i * loss
            np.maximum(max_loss_feat, loss, out=max_loss_feat)

            sum_loss_for_csl += loss

            if have_grad:
                if grad_path.exists():
                    g = load_grad_scalar(str(grad_path))
                    sum_grad += g
                    sumxy_grad += i * g
                    np.maximum(max_grad, g, out=max_grad)
                else:
                    have_grad = False

            w_count += 1

        if w_count == 0:
            continue

        if w_count != W:
            Sx_w = w_count*(w_count+1)/2.0
            Sx2_w = w_count*(w_count+1)*(2*w_count+1)/6.0
            denom = (w_count*Sx2_w - Sx_w*Sx_w)
        else:
            Sx_w, Sx2_w, denom = Sx, Sx2, denomW

        mean_curv = sum_curv / w_count
        slope_curv = (w_count*sumxy_curv - Sx_w*sum_curv) / (denom if denom != 0 else np.inf)

        mean_loss = sum_loss_feat / w_count
        slope_loss = (w_count*sumxy_loss_feat - Sx_w*sum_loss_feat) / (denom if denom != 0 else np.inf)

        if have_grad:
            mean_grad = sum_grad / w_count
            slope_grad = (w_count*sumxy_grad - Sx_w*sum_grad) / (denom if denom != 0 else np.inf)
        else:
            mean_grad = slope_grad = None

        csl_to_boundary = csl_prefix + sum_loss_for_csl

        end_loss_path = Path(args.loss_dir) / f'epoch_{end_e:04d}' / 'per_sample_losses.npy'
        next_loss_path = Path(args.loss_dir) / f'epoch_{end_e+1:04d}' / 'per_sample_losses.npy'
        if not end_loss_path.exists() or not next_loss_path.exists():
            csl_prefix = csl_to_boundary
            continue

        L_end = np_load(str(end_loss_path))
        L_next = np_load(str(next_loss_path))
        delta = L_next - L_end

        pm = (sample_intro_task <= t) & np.isfinite(delta)
        if pm.sum() == 0:
            csl_prefix = csl_to_boundary
            continue

        r = dict(
            task=t,
            n_prev_samples=int(pm.sum()),
            spearman_delta_curv_mean=float(spearmanr_np(delta[pm], mean_curv[pm])),
            spearman_delta_curv_max=float(spearmanr_np(delta[pm], max_curv[pm])),
            spearman_delta_curv_slope=float(spearmanr_np(delta[pm], slope_curv[pm])),
            spearman_delta_csl_to_boundary=float(spearmanr_np(delta[pm], csl_to_boundary[pm])),
            spearman_delta_loss_mean=float(spearmanr_np(delta[pm], mean_loss[pm])),
            spearman_delta_loss_max=float(spearmanr_np(delta[pm], max_loss_feat[pm])),
            spearman_delta_loss_slope=float(spearmanr_np(delta[pm], slope_loss[pm])),
        )
        if have_grad:
            r.update(
                spearman_delta_grad_mean=float(spearmanr_np(delta[pm], mean_grad[pm])),
                spearman_delta_grad_max=float(spearmanr_np(delta[pm], max_grad[pm])),
                spearman_delta_grad_slope=float(spearmanr_np(delta[pm], slope_grad[pm])),
            )
        rows.append(r)

        qs = np.quantile(mean_curv[pm], np.linspace(0, 1, 6))
        for j in range(len(qs)-1):
            low, high = qs[j], qs[j+1]
            sel = pm & (mean_curv >= low) & (mean_curv <= high)
            if sel.sum() > 0:
                qrows.append(dict(
                    task=t, qbin=j,
                    curv_mean_low=float(low), curv_mean_high=float(high),
                    delta_loss_mean=float(delta[sel].mean()),
                    delta_loss_median=float(np.median(delta[sel])),
                    n=int(sel.sum())
                ))

        csl_prefix = csl_to_boundary

    import csv
    if rows:
        with open(os.path.join(args.out_dir, 'plasticity_correlations.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if qrows:
        with open(os.path.join(args.out_dir, 'delta_loss_by_curv_quantile.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(qrows[0].keys()))
            w.writeheader()
            w.writerows(qrows)

    summary = dict(
        tasks_processed=len(rows),
        epochs_per_task=args.epochs_per_task,
        classes_per_task=args.classes_per_task,
        n_samples=int(n),
        out_dir=args.out_dir
    )
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
```

How to run (paths match your jobs):
```bash
python analyze_plasticity_curvature_csl.py \
  --loss_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_full \
  --curv_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_image_curvature_full \
  --class_order_file /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.npy \
  --epochs_per_task 200 \
  --classes_per_task 5 \
  --out_dir /scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/analysis/plasticity
```

Next tasks: relate network plasticity metrics to per-sample hardness
- Dormant neuron proportion across boundaries:
  - For each boundary t, evaluate a held-out set with checkpoints at e=t·200 and e=t·200+1.
  - For each layer, compute fraction of units with mean activation < ε (e.g., 1e-3) or active rate < p (e.g., <1% nonzero after ReLU).
  - Associate Δ(dormant proportion) with task-t aggregates of sample hardness (e.g., 90th percentile of curvature, CSL) and with gradient features (mean/max/slope).

- Growing weight magnitude across boundaries:
  - Load weights at e=t·200 and e=t·200+1; compute per-layer Δ||W||2 and mean |ΔW|.
  - Regress these changes on task-t aggregates of curvature/CSL/loss/grad features to see which best explain parameter growth.

- Joint boundary model:
  - Per-task OLS/GLM: Δloss ~ curvature_mean + CSL + loss_mean + grad_mean + interactions.
  - Repeat at per-class aggregation (mean/quantiles per class) to predict class-level forgetting.

- Ablation (optional):
  - Mask top-q percentile “hard” samples (by curvature/CSL) in the last epochs of task t; retrain across the boundary and measure changes in Δloss, Δ(dormant proportion), and Δ||W||.

Notes and interpretation tips:
- If curvature features (mean/max/slope) are predictive, you should see positive Spearman correlations with Δloss at most boundaries. CSL-to-boundary often captures “hardness”; curvature tends to capture sensitivity/instability. Their interaction is often strongest: high-curvature, high-CSL samples typically incur the largest plasticity loss.
- Use `delta_loss_by_curv_quantile.csv` to visualize Δloss across curvature quantiles per task and check for monotonicity.

- If you also have accuracy per task, we can extend this to predict class-level forgetting by aggregating these sample features per class before and after each boundary.

- Assumptions: 5 classes/task (CIFAR-100 over 20 tasks). Adjust `--classes_per_task` if different.

- This is streaming and safe: ~1.6–2.0 GB read total, no large arrays kept in memory.

- Optional next step: add a simple regression (e.g., Δloss ~ curvature_mean + CSL + interaction) per task to quantify marginal effects and get confidence intervals.

- Quick sanity checks to run before trusting results:
  - `summary.json` reports tasks processed and sample count.
  - Spot-check a boundary: confirm `delta` is positive on average for previous classes, near zero for new classes.

- If you want, I can add per-class correlation outputs or a couple of small plots.

- Status update: I reviewed your job outputs structure and stitched a streaming analysis that summarizes per-task curvature and CSL and correlates them with boundary loss jumps (plasticity). The script is ready to run and will emit CSVs and a summary JSON.

- Key outputs:
  - `plasticity_correlations.csv` (per-task Spearman correlations)
  - `delta_loss_by_curv_quantile.csv` (Δloss by curvature quantiles)
  - `summary.json` (run metadata)



- observation
    - easy sample is more clean and high memo score sample is more noisy.
    - low memo sample can reach to zero loss, but high mem sample can't, it is more obvious for gradient norm.
    - input curvature is the same.
    - all the csl, csg, csc, as well as, cse show the scale up for the hard samples.
    - sample definitely itself is a cause of everything, but the collection of sample will be.
    - can I separate catastrophic forgetting from plasticity loss?
    - what is the meening of non zero loss (or high loss) for perticular sample?
        - It means the model can't predict for particular sample.
        


    - 
- next
    - memorization score distribution for each epoch, track the sample id
    - for the order of class introduction, if it is a matter? I need more experiments
    - which statisics should I use to distinguish?
    - answer this questions: high memo sample what are you doing?
    - find and distinguish high and low memeo sample with a threshold like top 10, 20, 30 percent high memo sample and see their statistics? mean...

    - difference of catastrophic forgetting, which is more direct with the loss....
    - 

