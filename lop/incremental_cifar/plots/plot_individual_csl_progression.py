# built-in
import os
import argparse

# third party libraries
import matplotlib.pyplot as plt
import numpy as np


def get_colors(experiments: list):
    """
    Returns a dictionary of colors for different experiments/algorithms.
    """
    pre_assigned_colors = {
        "base_deep_learning_system": "#d62728",     # tab: red
        "retrained_network": "#7f7f7f",             # tab: grey
        "head_resetting": "#2ca02c",                # tab: green
        "shrink_and_perturb": "#ff7f0e",            # tab: orange
        "continual_backpropagation": "#1f77b4",     # tab: blue
    }
    return pre_assigned_colors.get(experiments[0], "#d62728")


def plot_individual_csl_progression(experiment: str, results_dir: str, dataset: str = "cifar100", 
                                   max_samples: int = 1000, alpha: float = 0.1, 
                                   show_mean: bool = True):
    """
    Plot individual CSL progression lines for each sample.
    
    :param experiment: experiment name
    :param results_dir: path to directory containing experiment results
    :param dataset: dataset name (default: cifar100)
    :param max_samples: maximum number of sample lines to plot
    :param alpha: transparency for individual lines
    :param show_mean: whether to show the mean progression line
    """
    
    per_sample_dir = os.path.join(results_dir, experiment, "per_sample_losses_inc")
    
    if not os.path.exists(per_sample_dir):
        print(f"Warning: Per-sample directory not found: {per_sample_dir}")
        return
    
    # Get available epoch files
    loss_files = [f for f in os.listdir(per_sample_dir) if f.startswith('loss_cifar100_epoch_')]
    
    if not loss_files:
        print(f"Warning: No loss files found in {per_sample_dir}")
        return
    
    # Extract epoch numbers and sort
    epochs = sorted([int(f.split('_')[-1].split('.')[0]) for f in loss_files])
    print(f"Found {len(epochs)} epochs for {experiment}: {epochs[0]} to {epochs[-1]}")
    
    # Load first file to determine number of samples
    first_loss_file = os.path.join(per_sample_dir, f'loss_cifar100_epoch_{epochs[0]}.npy')
    first_data = np.load(first_loss_file)
    num_total_samples = len(first_data)
    
    # Limit samples for visualization
    if max_samples is not None and max_samples < num_total_samples:
        sample_indices = np.random.choice(num_total_samples, max_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # Keep sorted for consistency
        num_samples = max_samples
    else:
        sample_indices = np.arange(num_total_samples)
        num_samples = num_total_samples
    
    print(f"Plotting {num_samples} individual CSL progression lines out of {num_total_samples} total samples")
    
    # Load all epoch data
    loss_arrays = []
    for epoch in epochs:
        loss_file = os.path.join(per_sample_dir, f'loss_cifar100_epoch_{epoch}.npy')
        if os.path.exists(loss_file):
            loss_data = np.load(loss_file)[sample_indices]
            loss_arrays.append(loss_data)
        else:
            print(f"Warning: Missing loss file for epoch {epoch}")
    
    if not loss_arrays:
        print("No loss data loaded")
        return
    
    # Compute CSL progression for each sample
    loss_matrix = np.array(loss_arrays)  # Shape: (num_epochs, num_samples)
    csl_progression = np.cumsum(loss_matrix, axis=0)  # Cumulative sum across epochs
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get color for this experiment
    color = get_colors([experiment])
    
    # Plot individual sample lines
    epochs_array = np.array(epochs[:len(csl_progression)])
    
    print(f"Plotting {csl_progression.shape[1]} individual progression lines...")
    
    # Plot each sample's CSL progression
    for sample_idx in range(csl_progression.shape[1]):
        ax.plot(epochs_array, csl_progression[:, sample_idx], 
               color=color, alpha=alpha, linewidth=0.5)
    
    # Optionally overlay the mean progression
    if show_mean:
        mean_progression = np.mean(csl_progression, axis=1)
        ax.plot(epochs_array, mean_progression, 
               color='black', linewidth=3, label=f'Mean CSL ({experiment})')
        ax.legend()
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cumulative Sample Loss (CSL)", fontsize=12)
    ax.set_title(f"Individual CSL Progression Lines\n{experiment} ({num_samples:,} samples)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    final_csl = csl_progression[-1, :]  # Final CSL values for all samples
    stats_text = f'Final CSL Statistics:\nMin: {final_csl.min():.1f}\nMax: {final_csl.max():.1f}\nMean: {final_csl.mean():.1f}\nStd: {final_csl.std():.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def plot_sample_subset_comparison(experiment: str, results_dir: str, dataset: str = "cifar100",
                                 sample_groups: list = [10, 100, 1000]):
    """
    Compare CSL progression for different numbers of sample lines.
    """
    
    per_sample_dir = os.path.join(results_dir, experiment, "per_sample_losses_inc")
    
    if not os.path.exists(per_sample_dir):
        print(f"Warning: Per-sample directory not found: {per_sample_dir}")
        return None
    
    # Get available epoch files
    loss_files = [f for f in os.listdir(per_sample_dir) if f.startswith('loss_cifar100_epoch_')]
    
    if not loss_files:
        print(f"Warning: No loss files found in {per_sample_dir}")
        return None
    
    fig, axes = plt.subplots(1, len(sample_groups), figsize=(5*len(sample_groups), 6))
    if len(sample_groups) == 1:
        axes = [axes]
    
    for i, num_samples in enumerate(sample_groups):
        # Get epochs
        epochs = sorted([int(f.split('_')[-1].split('.')[0]) for f in loss_files])
        
        # Load data for this subset
        first_loss_file = os.path.join(per_sample_dir, f'loss_cifar100_epoch_{epochs[0]}.npy')
        first_data = np.load(first_loss_file)
        
        # Sample indices
        sample_indices = np.random.choice(len(first_data), num_samples, replace=False)
        
        # Load and compute CSL progression
        loss_arrays = []
        for epoch in epochs:
            loss_file = os.path.join(per_sample_dir, f'loss_cifar100_epoch_{epoch}.npy')
            loss_data = np.load(loss_file)[sample_indices]
            loss_arrays.append(loss_data)
        
        loss_matrix = np.array(loss_arrays)
        csl_progression = np.cumsum(loss_matrix, axis=0)
        
        # Plot
        color = get_colors([experiment])
        alpha = max(0.1, min(1.0, 10.0 / num_samples))  # Adaptive alpha
        
        epochs_array = np.array(epochs)
        for sample_idx in range(csl_progression.shape[1]):
            axes[i].plot(epochs_array, csl_progression[:, sample_idx], 
                        color=color, alpha=alpha, linewidth=0.5)
        
        # Mean line
        mean_progression = np.mean(csl_progression, axis=1)
        axes[i].plot(epochs_array, mean_progression, 
                    color='black', linewidth=2, label='Mean')
        
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("CSL" if i == 0 else "")
        axes[i].set_title(f"{num_samples:,} Sample Lines")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    fig.suptitle(f"CSL Progression Comparison: {experiment}", fontsize=16)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot individual CSL progression lines for each sample")
    
    # Default to the results2 directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(os.path.dirname(script_dir), "results2")
    
    parser.add_argument("--results_dir", action="store", type=str, default=default_results_dir,
                        help="Path to directory containing the results of experiments.")
    parser.add_argument("--experiment", action="store", type=str, default="base_deep_learning_system",
                        help="Experiment name.")
    parser.add_argument("--max_samples", action="store", type=int, default=1000,
                        help="Maximum number of sample lines to plot (default: 1000)")
    parser.add_argument("--alpha", action="store", type=float, default=0.1,
                        help="Transparency for individual lines (default: 0.1)")
    parser.add_argument("--show_mean", action="store_true", default=True,
                        help="Show mean progression line")
    parser.add_argument("--comparison", action="store_true", 
                        help="Create comparison plot with different sample counts")
    parser.add_argument("--dataset", action="store", type=str, default="cifar100",
                        help="Dataset name (default: cifar100)")
    
    args = parser.parse_args()
    
    # Convert to absolute path to avoid relative path issues
    results_dir = os.path.abspath(args.results_dir)
    print(f"Using results directory: {results_dir}")
    
    if args.comparison:
        # Create comparison plot
        fig = plot_sample_subset_comparison(args.experiment, results_dir, args.dataset)
        if fig is None:
            print("❌ Failed to create comparison plot")
            return
        output_file = f"individual_csl_comparison_{args.experiment}.png"
    else:
        # Create individual progression plot
        result = plot_individual_csl_progression(
            args.experiment, results_dir, args.dataset, 
            args.max_samples, args.alpha, args.show_mean
        )
        if result is None:
            print("❌ Failed to create progression plot. Check if the required data files exist.")
            return
        fig, ax = result
        output_file = f"individual_csl_progression_{args.experiment}_{args.max_samples}samples.png"
    
    # Save plot
    file_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(file_path, output_file)
    plt.savefig(full_output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved plot: {full_output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
