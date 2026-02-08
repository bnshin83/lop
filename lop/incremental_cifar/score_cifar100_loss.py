#!/usr/bin/env python3
"""
CIFAR-100 Per-Sample Loss Extraction Script

This script computes per-sample cross-entropy loss values from CIFAR-100 model checkpoints.
These loss values are then used for proper CSL (Cumulative Sample Loss) calculation
according to the CSL-Mem paper definition.

Adapted from score_imagenet_loss.py
"""

import os
import torch
import numpy as np
import json
import argparse
import math
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.str2bool import str2bool
from utils.log_util import setup_logger
from scores import get_loss_for_batch, get_loss_and_grad_for_batch
import torch.nn as nn
import torch.nn.functional as F

# Define ResNet18 for CIFAR-100 (adapted from the project's architecture)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_CIFAR100():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)

def get_loss_and_grad_with_weight_decay(model, criterion, images, targets, weight_decay=1e-4):
    """
    Compute per-sample loss and gradients including weight decay regularization.
    This matches the loss function used during training.
    """
    # Forward pass
    outputs = model(images)
    
    # Cross-entropy loss per sample
    ce_losses = criterion(outputs, targets)  # Shape: (batch_size,)
    
    # Add L2 regularization (weight decay) to each sample's loss
    l2_reg = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.sum(param ** 2)
    
    # Total loss per sample (broadcast L2 reg to all samples)
    total_losses = ce_losses + weight_decay * l2_reg
    
    # Compute gradients w.r.t. total loss
    batch_size = images.size(0)
    gradients = torch.zeros(batch_size)
    
    for i in range(batch_size):
        # Zero gradients
        model.zero_grad()
        if images.grad is not None:
            images.grad.zero_()
        
        # Backward pass for individual sample
        total_losses[i].backward(retain_graph=True)
        
        # Compute gradient norm for this sample
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += torch.sum(param.grad ** 2)
        gradients[i] = torch.sqrt(grad_norm)
    
    return total_losses.detach(), gradients.detach(), None

class IndexedCIFAR100(datasets.CIFAR100):
    """CIFAR-100 dataset with sample indices."""
    
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return {
            'image': image,
            'label': target,
            'index': index
        }

def main():
    parser = argparse.ArgumentParser(description='Extract per-sample loss values from CIFAR-100 checkpoints',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Key parameters
    parser.add_argument('--run_idx',                default=1,              type=int,       help='Training run index (1, 2, etc.)')
    parser.add_argument('--noise_level',            default=0.01,           type=float,     help='Noise level (0.01, 0.02, 0.05, 0.1)')
    parser.add_argument('--start_epoch',            default=0,              type=int,       help='Starting epoch')
    parser.add_argument('--end_epoch',              default=199,            type=int,       help='Ending epoch')
    parser.add_argument('--epoch_step',             default=1,              type=int,       help='Epoch step size')
    parser.add_argument('--weight_decay',           default=1e-4,           type=float,     help='Weight decay used during training (default: 1e-4)')
    
    # Dataset parameters
    parser.add_argument('--dataset',                default='cifar100',     type=str,       help='Dataset name')
    parser.add_argument('--data_dir',               default='./data',       type=str,       help='Data directory')
    
    # Dataloader args
    parser.add_argument('--batch_size',             default=256,            type=int,       help='Batch size')
    parser.add_argument('--num_workers',            default=4,              type=int,       help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Use DataParallel')
    parser.add_argument('--checkpoint_dir',         default='./pretrained/cifar100/', type=str, help='Checkpoint directory')
    
    # Output parameters
    parser.add_argument('--output_dir',             default='./results/per_sample_losses/', type=str, help='Output directory for loss files')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    log_filename = f'score_cifar100_loss_run{args.run_idx}_noise{args.noise_level}.log'
    logger = setup_logger(logfile_name=log_filename)
    logger.info(f"Starting CIFAR-100 loss extraction with args: {args}")
    
    # Load CIFAR-100 dataset
    print("Loading CIFAR-100 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = IndexedCIFAR100(
        root=args.data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Important: don't shuffle to maintain index order
        num_workers=args.num_workers
    )
    
    dataset_len = len(train_dataset)  # Should be 50,000 for CIFAR-100
    logger.info(f"CIFAR-100 training set size: {dataset_len}")
    
    # Initialize model
    print("Initializing ResNet18 model...")
    net = ResNet18_CIFAR100()
    
    if args.parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net = torch.nn.DataParallel(net)
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    
    # Process each epoch
    epochs_to_process = range(args.start_epoch, args.end_epoch + 1, args.epoch_step)
    logger.info(f"Processing epochs: {list(epochs_to_process)}")
    
    for epoch in epochs_to_process:
        print(f"\n=== Processing Epoch {epoch} ===")
        logger.info(f'Processing epoch {epoch}')
        
        # Construct checkpoint filename
        checkpoint_filename = f"cifar100_resnet18_noisy_idx_{args.run_idx}_epoch_{epoch}_noise_{args.noise_level}.ckpt"
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            print(f"Skipping epoch {epoch} - checkpoint not found")
            continue
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_filename}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if args.parallel:
                # Copy the weight from cpu checkpoint to gpu
                net.module.load_state_dict(checkpoint)
            else:
                net.load_state_dict(checkpoint)
                
            logger.info(f'Successfully loaded checkpoint for epoch {epoch}')
        except Exception as e:
            logger.error(f'Failed to load checkpoint for epoch {epoch}: {str(e)}')
            print(f"Error loading checkpoint for epoch {epoch}: {str(e)}")
            continue
        # No dropout, running BN stats;
        net.eval()
        
        # Initialize storage for this epoch
        losses = torch.zeros(dataset_len)
        loss_grads = torch.zeros(dataset_len)
        
        # Compute per-sample losses and gradients
        print(f"Computing per-sample losses and gradients for epoch {epoch}...")
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = batch_data['image'].to(device)
            targets = batch_data['label'].to(device)
            indices = batch_data['index']
            
            # Enable gradients for gradient computation
            images.requires_grad_(True)
            net.zero_grad()
            
            # Compute both loss and gradients with weight decay
            batch_losses, batch_loss_grads, _ = get_loss_and_grad_with_weight_decay(
                net, criterion, images, targets, args.weight_decay
            )
            
            # Store at correct indices
            losses[indices] = batch_losses.cpu().detach()
            loss_grads[indices] = batch_loss_grads.cpu().detach()
            
            # Clean up gradients
            if images.grad is not None:
                images.grad.zero_()
        
        # Save both losses and gradients for this epoch
        # Loss file
        loss_filename = f"loss_cifar100_run{args.run_idx}_epoch_{epoch}_noise_{args.noise_level}.npy"
        loss_path = os.path.join(args.output_dir, loss_filename)
        np.save(loss_path, losses.numpy())
        logger.info(f'Saved losses for epoch {epoch} to {loss_path}')
        
        # Gradient file  
        loss_grad_filename = f"loss_grad_cifar100_run{args.run_idx}_epoch_{epoch}_noise_{args.noise_level}.npy"
        loss_grad_path = os.path.join(args.output_dir, loss_grad_filename)
        np.save(loss_grad_path, loss_grads.numpy())
        logger.info(f'Saved loss gradients for epoch {epoch} to {loss_grad_path}')
        
        print(f"Saved: {loss_filename} and {loss_grad_filename}")
        print(f"Loss stats - Mean: {losses.mean():.4f}, Std: {losses.std():.4f}, Min: {losses.min():.4f}, Max: {losses.max():.4f}")
        print(f"Grad stats - Mean: {loss_grads.mean():.4f}, Std: {loss_grads.std():.4f}, Min: {loss_grads.min():.4f}, Max: {loss_grads.max():.4f}")
    
    logger.info("CIFAR-100 loss and gradient extraction completed!")
    print("\n=== Loss and gradient extraction completed! ===")
    print(f"Loss and gradient files saved in: {args.output_dir}")
    print("Files generated:")
    print("  - loss_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy")
    print("  - loss_grad_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise_level}.npy")

if __name__ == '__main__':
    main()
