#!/usr/bin/env python3
"""
Re-calculate test accuracy using memo score filtering for existing experiment results.
This script loads trained models and applies the same memo filtering to test data
that was used for training data.

Supports both single-GPU and multi-GPU execution:
- Single GPU: python recalculate_test_accuracy_with_memo_filter.py --results_dir ... --memo_percent 80
- Multi GPU: torchrun --nproc_per_node=8 recalculate_test_accuracy_with_memo_filter.py --results_dir ... --memo_percent 80 --multi_gpu
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import necessary modules
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize
from lop.nets.torchvision_modified_resnet import build_resnet18

def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # SLURM environment
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # Alternative SLURM setup
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        # Torchrun environment
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    return rank, world_size, local_rank, device

def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def assign_epochs_to_gpu(checkpoint_epochs, world_size, rank):
    """Assign epochs to GPUs in round-robin fashion."""
    my_epochs = []
    for i, epoch in enumerate(checkpoint_epochs):
        if i % world_size == rank:
            my_epochs.append(epoch)
    return my_epochs

def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """Sub-samples the CIFAR 100 data set according to the given indices"""
    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()

def calculate_threshold_from_percent(memo_percent):
    """Calculate memo threshold from percentile."""
    sample_map_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
    sample_df = pd.read_csv(sample_map_path)
    threshold = np.percentile(sample_df['memorization_score'], memo_percent)
    print(f"Calculated {memo_percent}th percentile threshold: {threshold:.6f}")
    return threshold

def load_low_memo_samples(memo_percent):
    """Load sample IDs with memorization scores below the threshold."""
    threshold = calculate_threshold_from_percent(memo_percent)
    sample_map_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
    sample_df = pd.read_csv(sample_map_path)
    
    # Filter for low memo samples (below threshold)
    low_memo_mask = sample_df['memorization_score'] < threshold
    low_memo_samples = sample_df[low_memo_mask]['sample_id'].values
    
    print(f"Total samples: {len(sample_df)}")
    print(f"Low memo samples (< {threshold:.6f}): {len(low_memo_samples)} ({len(low_memo_samples)/len(sample_df)*100:.1f}%)")
    print(f"High memo samples (>= {threshold:.6f}): {len(sample_df) - len(low_memo_samples)} ({(len(sample_df) - len(low_memo_samples))/len(sample_df)*100:.1f}%)")
    
    return set(low_memo_samples), threshold

def filter_test_data_memo(cifar_data, memo_percent, random_seed=42):
    """Filter test data using the same memo percentage as training (random sampling)."""
    test_indices = torch.arange(len(cifar_data.data["data"]), dtype=torch.int32)
    
    # For test data, randomly sample the same percentage of samples to maintain consistency
    # memo_percent (e.g., 80) means we keep samples below 80th percentile = keep 80% of samples
    num_samples_to_keep = int(len(test_indices) * memo_percent / 100)
    if num_samples_to_keep == 0:
        print(f"Warning: No test samples to keep with {memo_percent}% filtering")
        return torch.tensor([], dtype=torch.int32)
    
    # Use deterministic sampling based on memo_percent for reproducibility
    torch.manual_seed(random_seed + int(memo_percent))
    sampled_indices = torch.randperm(len(test_indices))[:num_samples_to_keep]
    filtered_indices = test_indices[sampled_indices]
    print(f"Test data: randomly sampled {len(filtered_indices)} samples ({memo_percent}% of total)")
    return filtered_indices

def load_class_order():
    """Load the predetermined class order from the results directory."""
    class_order_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.csv"
    class_order_df = pd.read_csv(class_order_path)
    return class_order_df['Value'].values

def get_test_data(data_path, memo_percent, random_seed=42, use_all_data=False):
    """Load and filter test data."""
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=False,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),
        Normalize(mean=mean, std=std),
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))

    if use_all_data:
        print(f"Will use all test data for corresponding classes ({len(cifar_data.data['data'])} total samples)")
    else:
        # Apply memo filtering to test data
        filtered_test_indices = filter_test_data_memo(cifar_data, memo_percent, random_seed)
        
        if len(filtered_test_indices) > 0:
            subsample_cifar_data_set(sub_sample_indices=filtered_test_indices, cifar_data=cifar_data)
            print(f"Test data filtered to {len(filtered_test_indices)} low memo samples")
        else:
            print("Warning: No low memo samples found in test data")
            return None, None

    batch_size = 100
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=False, num_workers=1)
    return cifar_data, dataloader

def get_test_data_for_classes(data_path, current_classes, use_all_data=False, memo_percent=None, random_seed=42):
    """Load test data for specific classes, optionally with memo filtering."""
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=False,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),
        Normalize(mean=mean, std=std),
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))
    
    # Select current classes first
    cifar_data.select_new_partition(current_classes)
    
    if not use_all_data and memo_percent is not None:
        # Apply memo filtering to test data for current classes
        filtered_test_indices = filter_test_data_memo(cifar_data, memo_percent, random_seed)
        
        if len(filtered_test_indices) > 0:
            subsample_cifar_data_set(sub_sample_indices=filtered_test_indices, cifar_data=cifar_data)
        else:
            print(f"Warning: No low memo samples found for classes {current_classes[:5]}...")
            return None, None

    batch_size = 100
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=False, num_workers=1)
    return cifar_data, dataloader

def evaluate_network(net, test_dataloader, all_classes, current_num_classes, device):
    """Evaluate the network on filtered test data."""
    net.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    
    avg_loss = 0.0
    avg_acc = 0.0
    num_test_batches = 0
    
    with torch.no_grad():
        for _, sample in enumerate(test_dataloader):
            images = sample["image"].to(device)
            test_labels = sample["label"].to(device)
            test_predictions = net.forward(images)[:, all_classes[:current_num_classes]]

            avg_loss += loss_fn(test_predictions, test_labels)
            avg_acc += torch.mean((test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))
            num_test_batches += 1

    return avg_loss / num_test_batches, avg_acc / num_test_batches

def recalculate_test_accuracy(results_dir, memo_percent, run_index=0, data_path=None, gpu_id=0, use_all_data=False, multi_gpu=False):
    """Re-calculate test accuracy with memo filtering for all epochs."""
    
    # Setup device and distributed environment
    if multi_gpu:
        rank, world_size, local_rank, device = setup_distributed()
        if rank == 0:
            print(f"Multi-GPU mode: Using {world_size} GPUs")
    else:
        rank, world_size, local_rank = 0, 1, 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Single GPU mode: Using GPU {gpu_id}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    
    # Set default data path
    if data_path is None:
        file_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(file_path, "data")
    
    # Load class order
    all_classes = load_class_order()
    
    # Find all model checkpoints
    model_params_dir = os.path.join(results_dir, "model_parameters")
    checkpoint_files = [f for f in os.listdir(model_params_dir) 
                       if f.startswith(f"checkpoint_{memo_percent}pct_index-{run_index}") and f.endswith('.pt')]
    
    # Extract epoch numbers and sort
    checkpoint_epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_epoch-')[1].split('.pt')[0])
            checkpoint_epochs.append(epoch)
        except:
            continue
    
    checkpoint_epochs.sort()
    
    # Distribute epochs across GPUs
    if multi_gpu:
        my_epochs = assign_epochs_to_gpu(checkpoint_epochs, world_size, rank)
        if rank == 0:
            print(f"Found {len(checkpoint_epochs)} model checkpoints")
            print(f"GPU {rank} processing {len(my_epochs)} epochs")
    else:
        my_epochs = checkpoint_epochs
        print(f"Found {len(checkpoint_epochs)} model checkpoints")
    
    # Initialize network
    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    
    # Store results for this GPU
    my_test_accuracies = {}
    my_test_losses = {}
    
    if not use_all_data and not multi_gpu:
        # Load test data once with memo filtering (single GPU approach)
        if rank == 0:
            print("Loading and filtering test data...")
        test_data, test_dataloader = get_test_data(data_path, memo_percent, use_all_data=use_all_data)
        if test_data is None:
            if rank == 0:
                print("Failed to load test data")
            return
    
    # Process assigned epochs
    epochs_iter = tqdm(my_epochs, desc=f"GPU {rank} processing epochs") if rank == 0 else my_epochs
    for epoch in epochs_iter:
        # Load model checkpoint
        checkpoint_path = os.path.join(model_params_dir, f"checkpoint_{memo_percent}pct_index-{run_index}_epoch-{epoch}.pt")
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        # Determine current number of classes based on epoch
        current_num_classes = min(5 + (epoch // 200) * 5, 100)
        current_classes = all_classes[:current_num_classes]
        
        if use_all_data or multi_gpu:
            # Load test data for current classes (all samples for these classes or for multi-GPU)
            test_data, test_dataloader = get_test_data_for_classes(
                data_path, current_classes, use_all_data=use_all_data
            )
            if test_data is None:
                if rank == 0:
                    print(f"Failed to load test data for epoch {epoch}")
                continue
        else:
            # Select current classes for pre-filtered test data (single GPU only)
            test_data.select_new_partition(current_classes)
        
        # Evaluate on test data
        test_loss, test_accuracy = evaluate_network(net, test_dataloader, all_classes, current_num_classes, device)
        
        my_test_accuracies[epoch] = test_accuracy.cpu().numpy()
        my_test_losses[epoch] = test_loss.cpu().numpy()
        
        if epoch % 200 == 0 and rank == 0:  # Print progress at class addition points
            data_type = "all samples" if use_all_data else f"{memo_percent}% filtered samples"
            print(f"Epoch {epoch}: Test accuracy = {test_accuracy:.4f}, Num classes = {current_num_classes} ({data_type})")
    
    # Gather results from all GPUs if using multi-GPU
    if multi_gpu:
        all_accuracies = [None for _ in range(world_size)]
        all_losses = [None for _ in range(world_size)]
        
        dist.all_gather_object(all_accuracies, my_test_accuracies)
        dist.all_gather_object(all_losses, my_test_losses)
        
        if rank == 0:
            # Combine results from all GPUs
            combined_accuracies = {}
            combined_losses = {}
            
            for gpu_accuracies in all_accuracies:
                combined_accuracies.update(gpu_accuracies)
            
            for gpu_losses in all_losses:
                combined_losses.update(gpu_losses)
            
            # Sort by epoch
            sorted_epochs = sorted(combined_accuracies.keys())
            filtered_test_accuracies = [combined_accuracies[epoch] for epoch in sorted_epochs]
            filtered_test_losses = [combined_losses[epoch] for epoch in sorted_epochs]
        else:
            # Non-rank-0 processes don't save results
            return None, None
    else:
        # Single GPU: convert dict to list
        sorted_epochs = sorted(my_test_accuracies.keys())
        filtered_test_accuracies = [my_test_accuracies[epoch] for epoch in sorted_epochs]
        filtered_test_losses = [my_test_losses[epoch] for epoch in sorted_epochs]
    
    # Save results
    if use_all_data:
        output_dir = os.path.join(results_dir, f"all_data_experiment")
        file_suffix = f"all_data_index-{run_index}"
    else:
        output_dir = os.path.join(results_dir, f"low_memo_{memo_percent}pct_experiment")
        file_suffix = f"filtered_{memo_percent}pct_index-{run_index}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    test_acc_npy = os.path.join(output_dir, f"test_accuracy_{file_suffix}.npy")
    np.save(test_acc_npy, np.array(filtered_test_accuracies))
    
    test_loss_npy = os.path.join(output_dir, f"test_loss_{file_suffix}.npy")
    np.save(test_loss_npy, np.array(filtered_test_losses))
    
    # Save as CSV files
    test_acc_csv = os.path.join(output_dir, f"test_accuracy_{file_suffix}.csv")
    pd.DataFrame({'Value': filtered_test_accuracies}).to_csv(test_acc_csv, index=False)
    
    test_loss_csv = os.path.join(output_dir, f"test_loss_{file_suffix}.csv")
    pd.DataFrame({'Value': filtered_test_losses}).to_csv(test_loss_csv, index=False)
    
    data_type = "all test data" if use_all_data else f"filtered test data ({memo_percent}%)"
    print(f"Test accuracy results for {data_type} saved to:")
    print(f"  {test_acc_csv}")
    print(f"  {test_acc_npy}")
    print(f"Final test accuracy: {filtered_test_accuracies[-1]:.4f}")
    
    return filtered_test_accuracies, filtered_test_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-calculate test accuracy with memo filtering")
    parser.add_argument("--results_dir", "-r", type=str, required=True, 
                       help="Path to results directory (e.g., results_low_memo_80pct)")
    parser.add_argument("--memo_percent", "-p", type=float, required=True, 
                       help="Memo score percentile used in training (e.g., 80)")
    parser.add_argument("--run_index", "-i", type=int, default=0, help="Run index")
    parser.add_argument("--data_path", "-d", type=str, default=None, help="Path to CIFAR-100 data")
    parser.add_argument("--gpu_id", "-g", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--use_all_data", "-a", action="store_true", 
                       help="Use all test data without memo filtering")
    parser.add_argument("--multi_gpu", action="store_true", 
                       help="Use multi-GPU distributed processing")
    
    args = parser.parse_args()
    
    if args.use_all_data:
        print("Re-calculating test accuracy using all test samples for corresponding classes...")
    else:
        print(f"Re-calculating test accuracy with {args.memo_percent}% memo filtering...")
    print(f"Results directory: {args.results_dir}")
    
    try:
        recalculate_test_accuracy(
            results_dir=args.results_dir,
            memo_percent=args.memo_percent,
            run_index=args.run_index,
            data_path=args.data_path,
            gpu_id=args.gpu_id,
            use_all_data=args.use_all_data,
            multi_gpu=args.multi_gpu
        )
    finally:
        if args.multi_gpu:
            cleanup_distributed()