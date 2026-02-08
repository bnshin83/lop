#!/usr/bin/env python3
"""
Compute Cumulative Sample Loss (CSL) and learning-time (tau-based) from saved per-sample losses.

This script aggregates per-sample losses saved by score_cifar100_loss.py across epochs to produce:
- CSL per sample: sum over epochs of the per-sample loss
- Learning-time per sample: computed from running-average of per-sample loss gradients (if available)

It supports both file-naming schemes present in this repo:
- Gautschi format:  loss_{dataset}_epoch_{epoch}.npy, loss_grad_{dataset}_epoch_{epoch}.npy
- Gilbreth format:  loss_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise}.npy (and loss_grad_...)

Outputs are written to: {output_dir}/
- csl_{dataset}_run{run_idx}_noise_{noise}.npy (shape: [num_samples])
- lt_{dataset}_run{run_idx}_noise_{noise}.npy (shape: [num_samples], optional if grads missing)
- metrics_{dataset}_run{run_idx}_noise_{noise}.json (metadata including tau and epochs used)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np


def _epoch_file_paths(
    input_dir: str,
    dataset: str,
    run_idx: int,
    noise_level: float,
    epoch: int,
) -> Tuple[str, str]:
    """Return (loss_path, loss_grad_path) for the given epoch, trying both naming schemes."""
    # Gautschi format (preferred)
    loss_g = os.path.join(input_dir, f"loss_{dataset}_epoch_{epoch}.npy")
    grad_g = os.path.join(input_dir, f"loss_grad_{dataset}_epoch_{epoch}.npy")

    # Gilbreth (fallback)
    loss_b = os.path.join(
        input_dir, f"loss_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy"
    )
    grad_b = os.path.join(
        input_dir, f"loss_grad_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy"
    )

    loss_path = loss_g if os.path.exists(loss_g) else loss_b
    grad_path = grad_g if os.path.exists(grad_g) else grad_b
    return loss_path, grad_path


def load_per_sample_series(
    input_dir: str,
    dataset: str,
    run_idx: int,
    noise_level: float,
    start_epoch: int,
    end_epoch: int,
    epoch_step: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """Load per-epoch per-sample loss and (optionally) loss_grad arrays.

    Returns:
        losses:      shape (num_kept_epochs, num_samples)
        loss_grads:  shape (num_kept_epochs, num_samples) with per-epoch fallbacks to losses
                     when a grad file is missing, or None if no grad file found for any epoch
        kept_epochs: list of epochs actually loaded
    """
    losses: List[np.ndarray] = []
    grads: List[Optional[np.ndarray]] = []
    kept_epochs: List[int] = []
    any_grad_found = False

    for epoch in range(start_epoch, end_epoch + 1, epoch_step):
        loss_path, grad_path = _epoch_file_paths(input_dir, dataset, run_idx, noise_level, epoch)
        if not os.path.exists(loss_path):
            # Skip epochs without a loss file
            continue
        loss_arr = np.load(loss_path)
        losses.append(loss_arr)
        kept_epochs.append(epoch)

        if os.path.exists(grad_path):
            grads.append(np.load(grad_path))
            any_grad_found = True
        else:
            grads.append(None)

    if not losses:
        raise FileNotFoundError(
            f"No per-sample loss files found in {input_dir} for dataset={dataset}, "
            f"run_idx={run_idx}, noise={noise_level}."
        )

    losses_np = np.stack(losses, axis=0)  # (E, N)
    grads_np: Optional[np.ndarray]
    if any_grad_found:
        # Replace missing grad epochs with the corresponding loss epoch (proxy),
        # matching the notebook's behavior.
        filled_grads = []
        for g, l in zip(grads, losses):
            filled_grads.append(g if g is not None else l)
        grads_np = np.stack(filled_grads, axis=0)
    else:
        grads_np = None

    return losses_np, grads_np, kept_epochs


def compute_csl(losses: np.ndarray) -> np.ndarray:
    """CSL per-sample as sum of losses across epochs. losses shape: (E, N)."""
    return losses.sum(axis=0)


def compute_learning_time_from_grads(loss_grads: np.ndarray) -> Tuple[np.ndarray, float]:
    """Replicate the notebook's learning-time calculation from per-sample loss gradients.

    Steps:
      - learning_condition = running mean over epochs of per-sample loss_grad
      - tau = global mean of learning_condition
      - lt = 1 - mean(learning_condition < tau, axis=0)  in [0,1]
    """
    # Running average over epochs (E, N)
    e_count = np.arange(1, loss_grads.shape[0] + 1, dtype=np.float64)[:, None]
    learning_condition = np.cumsum(loss_grads, axis=0) / e_count
    tau = float(learning_condition.mean())
    below = learning_condition < tau
    lt = 1.0 - below.mean(axis=0)
    return lt.astype(np.float32), tau


def main():
    parser = argparse.ArgumentParser(
        description="Compute CSL and learning-time from saved per-sample losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, default="./results/per_sample_losses", help="Directory with per-epoch per-sample loss files")
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name used in file names")
    parser.add_argument("--run_idx", type=int, default=1, help="Run index used in file names (gilbreth format)")
    parser.add_argument("--noise_level", type=float, default=0.01, help="Noise level used in file names (gilbreth format)")
    parser.add_argument("--start_epoch", type=int, default=0, help="First epoch to include")
    parser.add_argument("--end_epoch", type=int, default=199, help="Last epoch to include (inclusive)")
    parser.add_argument("--epoch_step", type=int, default=1, help="Epoch step")
    parser.add_argument("--output_dir", type=str, default="./results/csl", help="Directory to save CSL outputs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading per-sample series from {args.input_dir} ...")
    losses, loss_grads, kept_epochs = load_per_sample_series(
        input_dir=args.input_dir,
        dataset=args.dataset,
        run_idx=args.run_idx,
        noise_level=args.noise_level,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        epoch_step=args.epoch_step,
    )
    print(f"  Loaded epochs: {len(kept_epochs)} -> {kept_epochs[:5]}{' ...' if len(kept_epochs) > 5 else ''}")
    print(f"  losses shape: {losses.shape}")
    if loss_grads is not None:
        print(f"  loss_grads shape: {loss_grads.shape}")
    else:
        print("  loss_grads: not found; will compute CSL only")

    # CSL
    csl = compute_csl(losses)
    csl_path = os.path.join(
        args.output_dir, f"csl_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.npy"
    )
    np.save(csl_path, csl.astype(np.float32))
    print(f"Saved CSL: {csl_path}  (shape: {csl.shape})")

    # Learning-time from grads (optional)
    lt_path = None
    tau = None
    if loss_grads is not None:
        lt, tau = compute_learning_time_from_grads(loss_grads)
        lt_path = os.path.join(
            args.output_dir, f"lt_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.npy"
        )
        np.save(lt_path, lt)
        print(f"Saved learning-time: {lt_path}  (shape: {lt.shape})  tau={tau:.6f}")

    # Metadata
    meta = {
        "dataset": args.dataset,
        "run_idx": args.run_idx,
        "noise_level": args.noise_level,
        "kept_epochs": kept_epochs,
        "num_epochs": len(kept_epochs),
        "losses_shape": list(losses.shape),
        "loss_grads_shape": list(loss_grads.shape) if loss_grads is not None else None,
        "csl_path": csl_path,
        "lt_path": lt_path,
        "tau": tau,
    }
    meta_path = os.path.join(
        args.output_dir, f"metrics_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.json"
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()


