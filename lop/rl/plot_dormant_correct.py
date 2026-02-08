#!/usr/bin/env python3
"""
Plot correct dormant units metric from existing .log files.
Uses the paper's definition: dormant = neurons with activity <= 1% of the time
"""
import os
import yaml
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_data(filepath):
    """Load data from .log file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compute_dormant_correct(pol_features_activity, threshold=0.01):
    """
    Compute correct dormant metric matching the original paper.

    Args:
        pol_features_activity: array of shape [timesteps, num_layers, h_dim]
            Each value is the fraction of time that neuron was active (0-1)
        threshold: activity threshold below which neuron is considered dormant

    Returns:
        dormant_pct: array of shape [timesteps] - % of neurons that are dormant
    """
    if hasattr(pol_features_activity, 'float'):
        pol_features_activity = pol_features_activity.float()
    if hasattr(pol_features_activity, 'numpy'):
        pol_features_activity = pol_features_activity.numpy()
    elif hasattr(pol_features_activity, 'cpu'):
        pol_features_activity = pol_features_activity.cpu().numpy()

    # Skip index 0 (pre-allocated zeros) and find where actual data ends
    # Data ends when all values become 0 (unfilled entries)
    pfa = pol_features_activity[1:]  # Skip index 0

    # Find last non-zero timestep
    row_sums = pfa.sum(axis=(1, 2))
    valid_mask = row_sums > 0
    if valid_mask.sum() == 0:
        return np.array([0])
    last_valid = np.where(valid_mask)[0][-1] + 1
    pfa = pfa[:last_valid]

    # For each timestep, compute fraction of neurons with activity <= threshold
    # Shape: [timesteps, num_layers, h_dim] -> [timesteps]
    dormant_pct = (pfa <= threshold).astype(float).mean(axis=(1, 2)) * 100
    return dormant_pct


def compute_dormant_wrong(pol_features_activity):
    """
    The WRONG metric that was being used (for comparison).
    This computes (1 - avg_activity) which is NOT the same as dormant neurons.
    """
    if hasattr(pol_features_activity, 'float'):
        pol_features_activity = pol_features_activity.float()
    if hasattr(pol_features_activity, 'numpy'):
        pol_features_activity = pol_features_activity.numpy()
    elif hasattr(pol_features_activity, 'cpu'):
        pol_features_activity = pol_features_activity.cpu().numpy()

    # Skip index 0 and find where actual data ends
    pfa = pol_features_activity[1:]
    row_sums = pfa.sum(axis=(1, 2))
    valid_mask = row_sums > 0
    if valid_mask.sum() == 0:
        return np.array([50])
    last_valid = np.where(valid_mask)[0][-1] + 1
    pfa = pfa[:last_valid]

    # Average activity across all neurons, then compute 1 - avg
    avg_activity = pfa.mean(axis=(1, 2))
    return (1 - avg_activity) * 100


def plot_config(cfg_path, seeds, color, label, ax, metric_fn, stride=100):
    """Plot dormant units for one config across seeds"""
    cfg = yaml.safe_load(open(cfg_path))
    data_dir = cfg['dir']

    all_dormant = []
    for seed in seeds:
        filepath = os.path.join(data_dir, f'{seed}.log')
        if not os.path.exists(filepath):
            print(f"  Skipping {filepath} (not found)")
            continue

        try:
            data = load_data(filepath)
            pfa = data.get('pol_features_activity')
            if pfa is None or len(pfa) == 0:
                print(f"  Skipping {filepath} (no pol_features_activity)")
                continue

            dormant = metric_fn(pfa)
            all_dormant.append(dormant)
            print(f"  Loaded {filepath}: {len(dormant)} timesteps, start={dormant[0]:.1f}%, end={dormant[-1]:.1f}%")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue

    if not all_dormant:
        print(f"  No data for {label}")
        return

    # Align lengths (truncate to shortest)
    min_len = min(len(d) for d in all_dormant)
    all_dormant = [d[:min_len] for d in all_dormant]
    all_dormant = np.array(all_dormant)

    # Compute mean and confidence interval
    mean = all_dormant.mean(axis=0)

    # Downsample for plotting
    steps = np.arange(0, len(mean) * 1000, 1000)  # steps in env interactions

    # Stride for smoother plots
    plot_idx = np.arange(0, len(mean), stride)
    plot_steps = steps[plot_idx] / 1e6  # Convert to millions
    plot_mean = mean[plot_idx]

    ax.plot(plot_steps, plot_mean, '-', linewidth=2, color=color, label=label)

    # Add confidence interval if multiple seeds
    if len(all_dormant) > 1:
        std = all_dormant.std(axis=0)
        plot_std = std[plot_idx]
        ax.fill_between(plot_steps, plot_mean - plot_std, plot_mean + plot_std,
                       alpha=0.2, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ant', choices=['ant', 'sant'],
                       help='Environment: ant or sant (slippery ant)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Seeds to include')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride for plotting (downsample)')
    parser.add_argument('--compare', action='store_true',
                       help='Also plot the wrong metric for comparison')
    args = parser.parse_args()

    # Config files
    cfg_dir = f'cfg/{args.env}'
    configs = [
        (f'{cfg_dir}/std.yml', 'C3', 'Standard PPO'),
        (f'{cfg_dir}/cbp.yml', 'C0', 'CBP'),
        (f'{cfg_dir}/ns.yml', 'C1', 'NS Adam'),
        (f'{cfg_dir}/l2.yml', 'C4', 'L2'),
    ]

    # Create figure
    if args.compare:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    print("=" * 60)
    print("Computing CORRECT dormant metric (activity <= 1%)")
    print("=" * 60)

    for cfg_path, color, label in configs:
        if not os.path.exists(cfg_path):
            print(f"Config not found: {cfg_path}")
            continue
        print(f"\n{label}:")
        plot_config(cfg_path, args.seeds, color, label, ax1,
                   compute_dormant_correct, args.stride)

    ax1.set_xlabel('Steps (millions)', fontsize=12)
    ax1.set_ylabel('Dormant Units (%)', fontsize=12)
    ax1.set_title(f'Correct Metric: % neurons with activity ≤ 1%\n({args.env.upper()})', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)

    if args.compare:
        print("\n" + "=" * 60)
        print("Computing WRONG dormant metric (1 - avg_activity) for comparison")
        print("=" * 60)

        for cfg_path, color, label in configs:
            if not os.path.exists(cfg_path):
                continue
            print(f"\n{label}:")
            plot_config(cfg_path, args.seeds, color, label, ax2,
                       compute_dormant_wrong, args.stride)

        ax2.set_xlabel('Steps (millions)', fontsize=12)
        ax2.set_ylabel('Dormant Units (%)', fontsize=12)
        ax2.set_title('Wrong Metric: 100% - avg activity\n(what was logged to WandB)', fontsize=14)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = f'dormant_units_{args.env}_correct.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
