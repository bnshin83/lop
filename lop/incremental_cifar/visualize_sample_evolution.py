#!/usr/bin/env python3
"""
Visualization of Loss, Gradient, Curvature, and Eigenvalue Evolution for Sample ID 8

This script creates comprehensive visualizations showing how various metrics evolve
for a specific sample (sample_id 8) over 4000 epochs of training.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_cifar100_class_name(class_id):
    """Get CIFAR-100 class name from class ID."""
    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    
    # Handle numpy types and NaN values
    try:
        class_id = int(class_id)
        if 0 <= class_id < len(cifar100_classes):
            return cifar100_classes[class_id]
        return f"class_{class_id}"
    except (ValueError, TypeError):
        return "unknown_class"

def load_sample_data(sample_id=8, max_epochs=4000):
    """Load data for a specific sample across all epochs."""
    
    loss_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_full"
    curvature_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_image_curvature_full"
    
    data = {
        'epochs': [],
        'losses': [],
        'gradients': [],
        'curvatures': [],
        'eigenvalues': [],
        'class_label': None,
        'memorization_score': None
    }
    
    print(f"Loading data for sample_id {sample_id} across {max_epochs} epochs...")
    
    # Load class label from pre-built CSV mapping
    map_file = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
    if os.path.exists(map_file):
        try:
            sample_class_df = pd.read_csv(map_file)
            sample_row = sample_class_df[sample_class_df['sample_id'] == sample_id]
            if len(sample_row) > 0:
                data['class_label'] = int(sample_row.iloc[0]['class_label'])
                data['memorization_score'] = float(sample_row.iloc[0]['memorization_score'])
                print(f"Found class label: {data['class_label']} ({get_cifar100_class_name(data['class_label'])})")
                print(f"Memorization score: {data['memorization_score']:.6f}")
            else:
                print(f"Sample {sample_id} not found in pre-built mapping")
        except Exception as e:
            print(f"Could not load class label mapping: {e}")
    else:
        print("Pre-built class mapping CSV not found. Run quick_sample_class_map.py first.")
    
    found_epochs = []
    
    # Check which epochs have data
    for epoch in range(1, max_epochs + 1):
        epoch_str = f"epoch_{epoch:04d}"
        loss_epoch_dir = os.path.join(loss_dir, epoch_str)
        curvature_epoch_dir = os.path.join(curvature_dir, epoch_str)
        
        if os.path.exists(loss_epoch_dir) and os.path.exists(curvature_epoch_dir):
            found_epochs.append(epoch)
    
    print(f"Found {len(found_epochs)} epochs with data")
    
    # Load data for each epoch
    for epoch in tqdm(found_epochs, desc="Loading epochs"):
        epoch_str = f"epoch_{epoch:04d}"
        
        try:
            # Load loss data
            loss_epoch_dir = os.path.join(loss_dir, epoch_str)
            ids_path = os.path.join(loss_epoch_dir, "per_sample_ids.csv")
            losses_path = os.path.join(loss_epoch_dir, "per_sample_losses.csv")
            grads_path = os.path.join(loss_epoch_dir, "per_sample_grads.csv")
            
            if os.path.exists(ids_path) and os.path.exists(losses_path):
                ids_df = pd.read_csv(ids_path)
                losses_df = pd.read_csv(losses_path)
                
                # Find the index of our sample_id
                sample_indices = ids_df[ids_df.iloc[:, 0] == sample_id].index
                
                if len(sample_indices) > 0:
                    idx = sample_indices[0]
                    loss_value = losses_df.iloc[idx].values[0] if len(losses_df.iloc[idx].values) > 0 else np.nan
                    
                    # Load gradient data
                    gradient_value = np.nan
                    if os.path.exists(grads_path):
                        grads_df = pd.read_csv(grads_path)
                        if idx < len(grads_df):
                            grad_vals = grads_df.iloc[idx].values
                            gradient_value = np.linalg.norm(grad_vals) if len(grad_vals) > 0 else np.nan
                    
                    # Load curvature data
                    curvature_epoch_dir = os.path.join(curvature_dir, epoch_str)
                    curvature_ids_path = os.path.join(curvature_epoch_dir, "per_image_sample_ids.csv")
                    curvature_path = os.path.join(curvature_epoch_dir, "per_image_curvature.csv")
                    eigenvalue_path = os.path.join(curvature_epoch_dir, "per_image_directional_eig.csv")
                    
                    curvature_value = np.nan
                    eigenvalue_value = np.nan
                    
                    if os.path.exists(curvature_ids_path) and os.path.exists(curvature_path):
                        curvature_ids_df = pd.read_csv(curvature_ids_path)
                        curvature_df = pd.read_csv(curvature_path)
                        
                        curvature_indices = curvature_ids_df[curvature_ids_df.iloc[:, 0] == sample_id].index
                        if len(curvature_indices) > 0:
                            curv_idx = curvature_indices[0]
                            curvature_value = curvature_df.iloc[curv_idx].values[0] if len(curvature_df.iloc[curv_idx].values) > 0 else np.nan
                            
                            # Load eigenvalue
                            if os.path.exists(eigenvalue_path):
                                eigenvalue_df = pd.read_csv(eigenvalue_path)
                                if curv_idx < len(eigenvalue_df):
                                    eigenvalue_value = eigenvalue_df.iloc[curv_idx].values[0] if len(eigenvalue_df.iloc[curv_idx].values) > 0 else np.nan
                    
                    # Store data
                    data['epochs'].append(epoch)
                    data['losses'].append(loss_value)
                    data['gradients'].append(gradient_value)
                    data['curvatures'].append(curvature_value)
                    data['eigenvalues'].append(eigenvalue_value)
        
        except Exception as e:
            print(f"Error loading epoch {epoch}: {e}")
            continue
    
    print(f"Successfully loaded data for {len(data['epochs'])} epochs")
    return data

def create_comprehensive_visualization(data, sample_id=8):
    """Create a comprehensive visualization of all metrics."""
    
    # Convert to numpy arrays for easier handling
    epochs = np.array(data['epochs'])
    losses = np.array(data['losses'])
    gradients = np.array(data['gradients'])
    curvatures = np.array(data['curvatures'])
    eigenvalues = np.array(data['eigenvalues'])
    
    # Remove NaN values for each metric
    valid_loss = ~np.isnan(losses)
    valid_grad = ~np.isnan(gradients)
    valid_curv = ~np.isnan(curvatures)
    valid_eig = ~np.isnan(eigenvalues)
    
    # Remove outlier initial values (first few epochs) - skip first 10 epochs
    skip_epochs = 10
    if len(epochs) > skip_epochs:
        epochs = epochs[skip_epochs:]
        losses = losses[skip_epochs:]
        gradients = gradients[skip_epochs:]
        curvatures = curvatures[skip_epochs:]
        eigenvalues = eigenvalues[skip_epochs:]
        
        # Recompute valid indices after skipping
        valid_loss = ~np.isnan(losses)
        valid_grad = ~np.isnan(gradients)
        valid_curv = ~np.isnan(curvatures)
        valid_eig = ~np.isnan(eigenvalues)
    
    print(f"Valid data points (after skipping first {skip_epochs} epochs) - Loss: {np.sum(valid_loss)}, Gradient: {np.sum(valid_grad)}, "
          f"Curvature: {np.sum(valid_curv)}, Eigenvalue: {np.sum(valid_eig)}")
    
    # Create figure with subplots - changed layout for cumulative plots
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Loss Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    if np.sum(valid_loss) > 0:
        ax1.plot(epochs[valid_loss], losses[valid_loss], color=colors[0], linewidth=0.5, alpha=0.8)
        ax1.scatter(epochs[valid_loss][::50], losses[valid_loss][::50], color=colors[0], s=30, alpha=0.6)
    ax1.set_title(f'Loss Evolution - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(0, 4000)
    ax1.set_xticks(np.arange(200, 4001, 200))
    ax1.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient Norm Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    if np.sum(valid_grad) > 0:
        ax2.plot(epochs[valid_grad], gradients[valid_grad], color=colors[1], linewidth=0.5, alpha=0.8)
        ax2.scatter(epochs[valid_grad][::50], gradients[valid_grad][::50], color=colors[1], s=30, alpha=0.6)
    ax2.set_title(f'Gradient Norm Evolution - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_xlim(0, 4000)
    ax2.set_xticks(np.arange(200, 4001, 200))
    ax2.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Curvature Evolution
    ax3 = fig.add_subplot(gs[1, 0])
    if np.sum(valid_curv) > 0:
        ax3.plot(epochs[valid_curv], curvatures[valid_curv], color=colors[2], linewidth=0.5, alpha=0.8)
        ax3.scatter(epochs[valid_curv][::50], curvatures[valid_curv][::50], color=colors[2], s=30, alpha=0.6)
    ax3.set_title(f'Curvature Evolution - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Curvature')
    ax3.set_xlim(0, 4000)
    ax3.set_xticks(np.arange(200, 4001, 200))
    ax3.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Eigenvalue Evolution
    ax4 = fig.add_subplot(gs[1, 1])
    if np.sum(valid_eig) > 0:
        ax4.plot(epochs[valid_eig], eigenvalues[valid_eig], color=colors[3], linewidth=0.5, alpha=0.8)
        ax4.scatter(epochs[valid_eig][::50], eigenvalues[valid_eig][::50], color=colors[3], s=30, alpha=0.6)
    ax4.set_title(f'Eigenvalue Evolution - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_xlim(0, 4000)
    ax4.set_xticks(np.arange(200, 4001, 200))
    ax4.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative Loss
    ax5 = fig.add_subplot(gs[2, 0])
    if np.sum(valid_loss) > 0:
        cumulative_loss = np.cumsum(losses[valid_loss])
        ax5.plot(epochs[valid_loss], cumulative_loss, color=colors[0], linewidth=0.5, alpha=0.8)
        ax5.scatter(epochs[valid_loss][::100], cumulative_loss[::100], color=colors[0], s=20, alpha=0.6)
    ax5.set_title(f'Cumulative Loss - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Cumulative Loss')
    ax5.set_xlim(0, 4000)
    ax5.set_xticks(np.arange(200, 4001, 200))
    ax5.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative Gradient Norm
    ax6 = fig.add_subplot(gs[2, 1])
    if np.sum(valid_grad) > 0:
        cumulative_grad = np.cumsum(gradients[valid_grad])
        ax6.plot(epochs[valid_grad], cumulative_grad, color=colors[1], linewidth=0.5, alpha=0.8)
        ax6.scatter(epochs[valid_grad][::100], cumulative_grad[::100], color=colors[1], s=20, alpha=0.6)
    ax6.set_title(f'Cumulative Gradient Norm - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Cumulative Gradient Norm')
    ax6.set_xlim(0, 4000)
    ax6.set_xticks(np.arange(200, 4001, 200))
    ax6.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative Curvature
    ax7 = fig.add_subplot(gs[3, 0])
    if np.sum(valid_curv) > 0:
        cumulative_curv = np.cumsum(curvatures[valid_curv])
        ax7.plot(epochs[valid_curv], cumulative_curv, color=colors[2], linewidth=0.5, alpha=0.8)
        ax7.scatter(epochs[valid_curv][::100], cumulative_curv[::100], color=colors[2], s=20, alpha=0.6)
    ax7.set_title(f'Cumulative Curvature - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Cumulative Curvature')
    ax7.set_xlim(0, 4000)
    ax7.set_xticks(np.arange(200, 4001, 200))
    ax7.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Cumulative Eigenvalue (absolute value for meaningful accumulation)
    ax8 = fig.add_subplot(gs[3, 1])
    if np.sum(valid_eig) > 0:
        cumulative_eig = np.cumsum(np.abs(eigenvalues[valid_eig]))
        ax8.plot(epochs[valid_eig], cumulative_eig, color=colors[3], linewidth=0.5, alpha=0.8)
        ax8.scatter(epochs[valid_eig][::100], cumulative_eig[::100], color=colors[3], s=20, alpha=0.6)
    ax8.set_title(f'Cumulative |Eigenvalue| - Sample ID {sample_id}', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Cumulative |Eigenvalue|')
    ax8.set_xlim(0, 4000)
    ax8.set_xticks(np.arange(200, 4001, 200))
    ax8.set_xticklabels([str(i) for i in range(2, 41, 2)])
    ax8.grid(True, alpha=0.3)
    
    # Add overall title with class and memorization information
    title = f'Training Dynamics Evolution for Sample ID {sample_id}'
    if data.get('class_label') is not None:
        class_name = get_cifar100_class_name(data['class_label'])
        title += f' - Class {data["class_label"]} ({class_name})'
    if data.get('memorization_score') is not None:
        title += f' - Memorization Score: {data["memorization_score"]:.3f}'
    title += f'\nOver {max(epochs)} Epochs (x-axis shows hundreds: 2=200, 4=400, etc.)'
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def create_statistical_summary(data, sample_id=8):
    """Create statistical summary plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['losses', 'gradients', 'curvatures', 'eigenvalues']
    titles = ['Loss', 'Gradient Norm', 'Curvature', 'Eigenvalue']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[i//2, i%2]
        values = np.array(data[metric])
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            # Histogram
            ax.hist(valid_values, bins=50, alpha=0.7, color=color, density=True)
            ax.axvline(np.mean(valid_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_values):.4f}')
            ax.axvline(np.median(valid_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_values):.4f}')
            
        ax.set_title(f'{title} Distribution - Sample ID {sample_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    title = f'Statistical Summary for Sample ID {sample_id}'
    if data.get('class_label') is not None:
        class_name = get_cifar100_class_name(data['class_label'])
        title += f' - Class {data["class_label"]} ({class_name})'
    if data.get('memorization_score') is not None:
        title += f' - Memorization Score: {data["memorization_score"]:.3f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    return fig

def main():
    """Main function to load data and create visualizations."""
    
    # List of sample IDs to visualize
    sample_ids = [135, 1477]  # Add more sample IDs as needed
    
    for sample_id in sample_ids:
        print(f"\n{'='*50}")
        print(f"Creating visualizations for sample_id {sample_id}")
        print(f"{'='*50}")
        
        # Load data
        data = load_sample_data(sample_id=sample_id, max_epochs=4000)
        
        if len(data['epochs']) == 0:
            print(f"No data found for sample_id {sample_id}")
            continue
        
        # Get class info for filename
        class_info = ""
        if data['class_label'] is not None:
            class_name = get_cifar100_class_name(data['class_label'])
            class_info = f"_class_{data['class_label']}_{class_name}"
        
        # Create comprehensive visualization
        fig1 = create_comprehensive_visualization(data, sample_id)
        output_file1 = f'/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/plot_curvature/sample_{sample_id}{class_info}_evolution.png'
        fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive visualization to: {output_file1}")
        plt.close(fig1)  # Close figure to free memory
        
        # Create statistical summary
        fig2 = create_statistical_summary(data, sample_id)
        output_file2 = f'/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/plot_curvature/sample_{sample_id}{class_info}_statistics.png'
        fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"Saved statistical summary to: {output_file2}")
        plt.close(fig2)  # Close figure to free memory
        
        # Print summary statistics
        print(f"\n=== Summary Statistics for Sample ID {sample_id} ===")
        if data['class_label'] is not None:
            print(f"Class: {data['class_label']} ({get_cifar100_class_name(data['class_label'])})")
        if data['memorization_score'] is not None:
            print(f"Memorization Score: {data['memorization_score']:.6f}")
        print(f"Total epochs with data: {len(data['epochs'])}")
        print(f"Epoch range: {min(data['epochs'])} - {max(data['epochs'])}")
        
        for metric_name, metric_data in [('Loss', data['losses']), ('Gradient', data['gradients']), 
                                         ('Curvature', data['curvatures']), ('Eigenvalue', data['eigenvalues'])]:
            values = np.array(metric_data)
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                print(f"\n{metric_name}:")
                print(f"  Valid data points: {len(valid_values)}")
                print(f"  Mean: {np.mean(valid_values):.6f}")
                print(f"  Std: {np.std(valid_values):.6f}")
                print(f"  Min: {np.min(valid_values):.6f}")
                print(f"  Max: {np.max(valid_values):.6f}")
    
    print(f"\n{'='*50}")
    print("All visualizations completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()