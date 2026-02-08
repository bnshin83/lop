#!/usr/bin/env python3
"""
Unified CSL pipeline:
- Option A: Generate per-sample loss/grad from incremental CIFAR checkpoints (model_parameters)
- Option B: Use an existing directory of per-sample loss/grad .npy files
Then aggregate to produce CSL and tau-based learning-time (matching the notebook).

Outputs:
- CSL:  csl_{dataset}_run{run_idx}_noise_{noise}.npy (run_idx/noise kept for compatibility)
- LT:   lt_{dataset}_run{run_idx}_noise_{noise}.npy
- Meta: metrics_{dataset}_run{run_idx}_noise_{noise}.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lop.nets.torchvision_modified_resnet import build_resnet18


class IndexedCIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return {"image": image, "label": target, "index": index}


def list_available_epochs(model_parameters_dir: str, experiment_index: int) -> List[int]:
    epochs: List[int] = []
    # Try standard format first: index-{experiment_index}_epoch-{epoch}.pt
    prefix = f"index-{experiment_index}_epoch-"
    if os.path.isdir(model_parameters_dir):
        for fname in os.listdir(model_parameters_dir):
            if fname.startswith(prefix) and fname.endswith(".pt"):
                try:
                    ep_str = fname[len(prefix):-3]
                    epochs.append(int(ep_str))
                except Exception:
                    pass
    
    # If no epochs found, try checkpoint format: checkpoint_index-{experiment_index}_epoch-{epoch}.pt
    if not epochs:
        checkpoint_prefix = f"checkpoint_index-{experiment_index}_epoch-"
        for fname in os.listdir(model_parameters_dir):
            if fname.startswith(checkpoint_prefix) and fname.endswith(".pt"):
                try:
                    ep_str = fname[len(checkpoint_prefix):-3]
                    epochs.append(int(ep_str))
                except Exception:
                    pass
    
    epochs.sort()
    return epochs


def get_loss_and_grad_for_batch(model: torch.nn.Module, criterion: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, targets)
    grad = torch.autograd.grad(loss.sum(), images)[0]
    loss_grad = grad.reshape(grad.size(0), -1).norm(dim=1).detach()
    model.zero_grad()
    if images.grad is not None:
        images.grad.zero_()
    return loss.detach(), loss_grad


def _epoch_file_paths(input_dir: str, dataset: str, run_idx: int, noise_level: float, epoch: int) -> Tuple[str, str]:
    # Gautschi format
    loss_g = os.path.join(input_dir, f"loss_{dataset}_epoch_{epoch}.npy")
    grad_g = os.path.join(input_dir, f"loss_grad_{dataset}_epoch_{epoch}.npy")
    # Gilbreth format
    loss_b = os.path.join(input_dir, f"loss_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy")
    grad_b = os.path.join(input_dir, f"loss_grad_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy")
    loss_path = loss_g if os.path.exists(loss_g) else loss_b
    grad_path = grad_g if os.path.exists(grad_g) else grad_b
    return loss_path, grad_path


def load_per_sample_series(input_dir: str, dataset: str, run_idx: int, noise_level: float, start_epoch: int, end_epoch: int, epoch_step: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    losses: List[np.ndarray] = []
    grads: List[Optional[np.ndarray]] = []
    kept_epochs: List[int] = []
    any_grad_found = False

    for epoch in range(start_epoch, end_epoch + 1, epoch_step):
        loss_path, grad_path = _epoch_file_paths(input_dir, dataset, run_idx, noise_level, epoch)
        if not os.path.exists(loss_path):
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
        raise FileNotFoundError(f"No per-sample loss files found in {input_dir}")

    losses_np = np.stack(losses, axis=0)
    if any_grad_found:
        filled = []
        for g, l in zip(grads, losses):
            filled.append(g if g is not None else l)
        grads_np: Optional[np.ndarray] = np.stack(filled, axis=0)
    else:
        grads_np = None
    return losses_np, grads_np, kept_epochs


def compute_csl(losses: np.ndarray) -> np.ndarray:
    return losses.sum(axis=0)


def compute_learning_time_from_grads(loss_grads: np.ndarray) -> Tuple[np.ndarray, float]:
    e_count = np.arange(1, loss_grads.shape[0] + 1, dtype=np.float64)[:, None]
    learning_condition = np.cumsum(loss_grads, axis=0) / e_count
    tau = float(learning_condition.mean())
    lt = 1.0 - (learning_condition < tau).mean(axis=0)
    return lt.astype(np.float32), tau


def generate_per_sample_from_model_params(results_dir: str, data_path: str, experiment_index: int, start_epoch: Optional[int], end_epoch: Optional[int], batch_size: int, num_workers: int, device: torch.device, per_sample_out_dir: str, dataset: str = "cifar100", compute_grad: bool = True) -> Tuple[int, int]:
    os.makedirs(per_sample_out_dir, exist_ok=True)
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    train_dataset = IndexedCIFAR100(root=data_path, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_len = len(train_dataset)

    # Model
    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Epochs from model_parameters (or directly from results_dir if no subdirectory)
    mp_dir = os.path.join(results_dir, "model_parameters")
    if not os.path.exists(mp_dir):
        # Checkpoints are directly in results_dir
        mp_dir = results_dir
    epochs = list_available_epochs(mp_dir, experiment_index)
    if not epochs:
        raise FileNotFoundError(f"No model parameter files found in {mp_dir}")
    if start_epoch is not None:
        epochs = [e for e in epochs if e >= start_epoch]
    if end_epoch is not None:
        epochs = [e for e in epochs if e <= end_epoch]

    for epoch in epochs:
        # Try standard format first
        ckpt_path = os.path.join(mp_dir, f"index-{experiment_index}_epoch-{epoch}.pt")
        if not os.path.exists(ckpt_path):
            # Try checkpoint format
            ckpt_path = os.path.join(mp_dir, f"checkpoint_index-{experiment_index}_epoch-{epoch}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Missing checkpoint for epoch {epoch}; skipping")
                continue
        print(f"Scoring epoch {epoch}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.to(device).eval()

        losses = torch.zeros(dataset_len, dtype=torch.float32)
        loss_grads = torch.zeros(dataset_len, dtype=torch.float32) if compute_grad else None

        with torch.set_grad_enabled(compute_grad):
            for batch in train_loader:
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                indices = batch["index"]
                if compute_grad:
                    batch_losses, batch_grads = get_loss_and_grad_for_batch(net, criterion, images, targets)
                    losses[indices] = batch_losses.detach().cpu()
                    loss_grads[indices] = batch_grads.detach().cpu()
                else:
                    batch_losses = criterion(net(images), targets)
                    losses[indices] = batch_losses.detach().cpu()

        np.save(os.path.join(per_sample_out_dir, f"loss_{dataset}_epoch_{epoch}.npy"), losses.numpy())
        if compute_grad and loss_grads is not None:
            np.save(os.path.join(per_sample_out_dir, f"loss_grad_{dataset}_epoch_{epoch}.npy"), loss_grads.numpy())

    return (epochs[0] if epochs else 0), (epochs[-1] if epochs else -1)


def main():
    parser = argparse.ArgumentParser(description="Unified CSL pipeline",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Source selection
    parser.add_argument("--source", type=str, default="per_samples", choices=["per_samples", "model_parameters"],
                        help="Where to get per-sample losses from")
    # When source=per_samples
    parser.add_argument("--input_dir", type=str, default=None, help="Dir with per-sample loss/grad .npy files")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--run_idx", type=int, default=1)
    parser.add_argument("--noise_level", type=float, default=0.01)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=199)
    parser.add_argument("--epoch_step", type=int, default=1)
    # When source=model_parameters
    parser.add_argument("--results_dir", type=str, default=None, help="Results dir containing model_parameters/")
    parser.add_argument("--data_path", type=str, default=None, help="CIFAR-100 data dir")
    parser.add_argument("--experiment_index", type=int, default=0)
    parser.add_argument("--per_sample_out_dir", type=str, default=None,
                        help="Where to write generated per-sample files (defaults under results_dir)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute_grad", type=str, default="true", choices=["true", "false"])
    # Outputs
    parser.add_argument("--output_dir", type=str, default="./results/csl")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # If generating from model_parameters, do that first
    if args.source == "model_parameters":
        assert args.results_dir and args.data_path, "results_dir and data_path are required for source=model_parameters"
        per_sample_out_dir = args.per_sample_out_dir or os.path.join(args.results_dir, "per_sample_losses_inc")
        print(f"Generating per-sample files into {per_sample_out_dir} ...")
        _ = generate_per_sample_from_model_params(
            results_dir=args.results_dir,
            data_path=args.data_path,
            experiment_index=args.experiment_index,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            per_sample_out_dir=per_sample_out_dir,
            dataset=args.dataset,
            compute_grad=(args.compute_grad.lower() == "true"),
        )
        input_dir = per_sample_out_dir
    else:
        assert args.input_dir, "input_dir is required for source=per_samples"
        input_dir = args.input_dir

    # Aggregate to CSL and LT
    print(f"Aggregating per-sample arrays in {input_dir} ...")
    losses, loss_grads, kept_epochs = load_per_sample_series(
        input_dir=input_dir,
        dataset=args.dataset,
        run_idx=args.run_idx,
        noise_level=args.noise_level,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        epoch_step=args.epoch_step,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    csl = compute_csl(losses)
    csl_path = os.path.join(args.output_dir, f"csl_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.npy")
    np.save(csl_path, csl.astype(np.float32))

    lt_path = None
    tau = None
    if loss_grads is not None:
        lt, tau = compute_learning_time_from_grads(loss_grads)
        lt_path = os.path.join(args.output_dir, f"lt_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.npy")
        np.save(lt_path, lt)

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
        "input_dir": input_dir,
        "source": args.source,
    }
    meta_path = os.path.join(args.output_dir, f"metrics_{args.dataset}_run{args.run_idx}_noise_{args.noise_level}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {csl_path} {('and ' + lt_path) if lt_path else ''}\nMetadata: {meta_path}")


if __name__ == "__main__":
    main()


