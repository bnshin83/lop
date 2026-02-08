#!/usr/bin/env python3
"""
Score per-sample loss and input-gradient norm for incremental CIFAR models.

This scorer is aligned with the incremental training setup used by
`incremental_cifar_experiment.py` and `post_run_analysis_modified2.py`:

- Loads model weights from `results/<experiment_name>/model_parameters/index-<exp>_epoch-<epoch>.pt`
- Uses the same architecture: `lop/nets/torchvision_modified_resnet.build_resnet18`
- Uses CIFAR-100 training set with the project's normalization
- Writes per-epoch per-sample arrays suitable for `compute_csl.py`

Notes:
- By default, we compute cross-entropy loss across all 100 classes to provide a consistent CSL baseline across epochs.
- If you need losses restricted to the classes seen so far in incremental training, that can be added as an option.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Ensure repository root is on PYTHONPATH so 'lop.*' imports resolve when running directly
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lop.nets.torchvision_modified_resnet import build_resnet18


class IndexPreservingSubset(Dataset):
    """Dataset wrapper that preserves original indices when subsetting."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices  # Original indices in the full dataset
    
    def __getitem__(self, idx):
        # Get the sample from the original dataset
        original_idx = self.indices[idx]
        sample = self.dataset[original_idx]
        
        # Add the original index to the sample
        if isinstance(sample, dict):
            sample['index'] = original_idx
        else:
            # Convert to dict format and add original index
            image, label = sample
            sample = {
                'image': image, 
                'label': label,
                'index': original_idx
            }
        
        return sample
    
    def __len__(self):
        return len(self.indices)


class IndexedCIFAR100(datasets.CIFAR100):
    """CIFAR-100 dataset that returns the global index alongside image and label."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return {
            "image": image,
            "label": target,
            "index": index,
        }


@torch.no_grad()
def get_loss_for_batch(model: torch.nn.Module, criterion: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    loss = criterion(outputs, targets)
    return loss


def get_loss_and_grad_for_batch(model: torch.nn.Module, criterion: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, targets)  # (B,)
    grad = torch.autograd.grad(loss.sum(), images)[0]
    loss_grad = grad.reshape(grad.size(0), -1).norm(dim=1).detach()
    model.zero_grad()
    if images.grad is not None:
        images.grad.zero_()
    return loss.detach(), loss_grad


def list_available_epochs(model_parameters_dir: str, experiment_index: int) -> List[int]:
    epochs: List[int] = []
    prefix = f"index-{experiment_index}_epoch-"
    if not os.path.isdir(model_parameters_dir):
        return epochs
    for fname in os.listdir(model_parameters_dir):
        if fname.startswith(prefix) and fname.endswith(".pt"):
            try:
                ep_str = fname[len(prefix):-3]
                epochs.append(int(ep_str))
            except Exception:
                pass
    epochs.sort()
    return epochs


def compute_active_classes_for_epoch(epoch: int, class_order: np.ndarray) -> List[int]:
    """
    Compute the active classes for a given epoch based on incremental learning schedule.
    
    Args:
        epoch: Training epoch (1-based)
        class_order: Array of class indices in training order (from class_order/index-0.npy)
    
    Returns:
        List of active class indices for this epoch
    """
    # Start with 5 classes, add 5 every epoch up to 100 classes
    current_num_classes = min(5 + max(0, (epoch - 1) * 5), 100)
    current_num_classes = min(current_num_classes, len(class_order))
    
    # Active classes = class_order[:current_num_classes]
    active_classes = class_order[:current_num_classes].tolist()
    
    return active_classes


def create_filtered_dataset(dataset: Dataset, active_classes: List[int]) -> Dataset:
    """Create dataset filtered to only include samples from active classes."""
    indices = []
    
    # Get labels - handle both IndexedCIFAR100 and regular dataset
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        # Fallback: iterate through dataset to get labels
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict):
                labels.append(sample['label'])
            else:
                labels.append(sample[1])  # (image, label) format
    
    active_classes_set = set(active_classes)
    
    for i, label in enumerate(labels):
        if label in active_classes_set:
            indices.append(i)
    
    if not indices:
        raise ValueError(f"No samples found for active_classes={active_classes}")
    
    return IndexPreservingSubset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description="Score per-sample loss/grad for incremental CIFAR checkpoints",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results dir (e.g., .../results/base_deep_learning_system)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CIFAR-100 data directory (same used by training)")
    parser.add_argument("--experiment_index", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=None, help="Optional: restrict epochs >= this")
    parser.add_argument("--end_epoch", type=int, default=None, help="Optional: restrict epochs <= this")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save per-sample arrays; default: <results_dir>/per_sample_losses_inc")
    parser.add_argument("--compute_grad", type=str, default="true", choices=["true", "false"],
                        help="Whether to compute input-gradient norms as well")

    args = parser.parse_args()

    results_dir = args.results_dir
    model_parameters_dir = os.path.join(results_dir, "model_parameters")
    output_dir = args.output_dir or os.path.join(results_dir, "per_sample_losses_inc")
    os.makedirs(output_dir, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data (match project normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    train_dataset = IndexedCIFAR100(root=args.data_path, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataset_len = len(train_dataset)  # 50,000

    # Model
    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Epochs
    all_epochs = list_available_epochs(model_parameters_dir, args.experiment_index)
    if not all_epochs:
        raise FileNotFoundError(f"No model parameter files found in {model_parameters_dir}")
    if args.start_epoch is not None:
        all_epochs = [e for e in all_epochs if e >= args.start_epoch]
    if args.end_epoch is not None:
        all_epochs = [e for e in all_epochs if e <= args.end_epoch]
    print(f"Scoring epochs: {all_epochs[:10]}{' ...' if len(all_epochs) > 10 else ''}")

    compute_grad = (args.compute_grad.lower() == "true")

    for epoch in all_epochs:
        ckpt_path = os.path.join(model_parameters_dir, f"index-{args.experiment_index}_epoch-{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint: {ckpt_path}; skipping")
            continue

        print(f"\n=== Epoch {epoch} ===")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        losses = torch.zeros(dataset_len, dtype=torch.float32)
        loss_grads = torch.zeros(dataset_len, dtype=torch.float32) if compute_grad else None

        with torch.set_grad_enabled(compute_grad):
            total = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                indices = batch["index"]  # These are now the original dataset indices

                if compute_grad:
                    batch_losses, batch_loss_grads = get_loss_and_grad_for_batch(net, criterion, images, targets)
                    losses[indices] = batch_losses.detach().cpu()
                    loss_grads[indices] = batch_loss_grads.detach().cpu()
                else:
                    batch_losses = get_loss_for_batch(net, criterion, images, targets)
                    losses[indices] = batch_losses.detach().cpu()
                total += images.size(0)

        # Extract only the non-zero entries (active samples) for saving
        # This matches the curvature analysis approach of saving only processed samples
        active_mask = losses > 0  # Only samples that were actually processed
        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        
        if len(active_indices) == 0:
            print(f"Warning: No samples processed for epoch {epoch}")
            continue
            
        # Save results for active samples only
        active_losses = losses[active_indices]
        
        # Create results similar to curvature analysis format
        results_dict = {
            'sample_ids': active_indices.numpy(),
            'losses': active_losses.numpy(),
            'labels': torch.tensor([base_train_dataset.targets[i] for i in active_indices]).numpy(),
            'epoch': epoch,
            'active_classes': active_classes,
            'total_samples': len(active_indices)
        }
        
        # Save in format compatible with compute_csl.py
        loss_filename = f"loss_cifar100_epoch_{epoch}.npy"
        np.save(os.path.join(output_dir, loss_filename), active_losses.numpy())
        
        # Save sample IDs for each loss value
        sample_ids_filename = f"sample_ids_cifar100_epoch_{epoch}.npy"
        np.save(os.path.join(output_dir, sample_ids_filename), active_indices.numpy())
        
        # Save labels for each sample
        labels_filename = f"labels_cifar100_epoch_{epoch}.npy"
        active_labels = torch.tensor([base_train_dataset.targets[i] for i in active_indices])
        np.save(os.path.join(output_dir, labels_filename), active_labels.numpy())
        
        if compute_grad and loss_grads is not None:
            active_loss_grads = loss_grads[active_indices]
            grad_filename = f"loss_grad_cifar100_epoch_{epoch}.npy"
            np.save(os.path.join(output_dir, grad_filename), active_loss_grads.numpy())
        
        # Save metadata for traceability (similar to curvature analysis)
        metadata_filename = f"metadata_epoch_{epoch}.npy"
        np.save(os.path.join(output_dir, metadata_filename), results_dict)
        
        print(f"Saved per-sample arrays for epoch {epoch}: {len(active_indices)} samples to {output_dir}")


if __name__ == "__main__":
    main()


