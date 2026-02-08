#!/usr/bin/env python3
"""Plot all metrics for Ant and SlipperyAnt experiments from log files."""

import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

# Style settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Rolling window size for variance computation
WINDOW_SIZE = 5

# Color scheme matching reference
COLORS = {
    'std': '#E24A33',    # Red - Standard PPO
    'cbp': '#348ABD',    # Blue - CBP
    'ns': '#FFA500',     # Orange - NS Adam
    'l2': '#988ED5',     # Purple - L2
}

LABELS = {
    'std': 'Standard PPO',
    'cbp': 'CBP',
    'ns': 'NS Adam',
    'l2': 'L2',
}

# Output log file paths
OUTPUT_LOGS = {
    'ant': {
        'std': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228737_ant_std.out',
        'cbp': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228714_ant_cbp.out',
        'ns': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228719_ant_ns.out',
        'l2': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228718_ant_l2.out',
    },
    'sant': {
        'std': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228738_sant_std.out',
        'cbp': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228724_sant_cbp.out',
        'ns': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228721_sant_ns.out',
        'l2': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs/5228722_sant_l2.out',
    }
}

# Pickle log file paths
PICKLE_LOGS = {
    'ant': {
        'std': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/ant/bp/std/0.log',
        'cbp': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/ant/cbp/0.log',
        'ns': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/ant/bp/ns/0.log',
        'l2': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/ant/l2/0.log',
    },
    'sant': {
        'std': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/sant/bp/std/0.log',
        'cbp': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/sant/cbp/0.log',
        'ns': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/sant/bp/ns/0.log',
        'l2': '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/data/sant/l2/0.log',
    }
}

# Total steps for each environment
TOTAL_STEPS = {
    'ant': 100_000_000,
    'sant': 20_000_000,
}


def rolling_mean_std(values, window=WINDOW_SIZE):
    """Compute rolling mean and standard deviation."""
    if len(values) < window:
        return values, np.zeros_like(values)

    # Rolling mean
    mean = uniform_filter1d(values.astype(float), size=window, mode='nearest')

    # Rolling std (compute variance first)
    values_sq = values.astype(float) ** 2
    mean_sq = uniform_filter1d(values_sq, size=window, mode='nearest')
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 0)  # Ensure non-negative
    std = np.sqrt(variance)

    return mean, std


def parse_output_returns(filepath):
    """Parse output log file for returns."""
    steps, returns = [], []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'Step (\d+)/\d+.*Avg Return \(last 100\): ([-\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                returns.append(float(match.group(2)))
    return np.array(steps), np.array(returns)


def parse_output_dormant(filepath):
    """Parse output log file for dormant units."""
    steps, dormant_pct = [], []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\[Update \d+\] Step (\d+): Actual Dead=\d+/\d+ \(([\d.]+)%\)', line)
            if match:
                steps.append(int(match.group(1)))
                dormant_pct.append(float(match.group(2)))
    return np.array(steps), np.array(dormant_pct)


def load_pickle_log(filepath):
    """Load pickle log file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_stable_rank(data, total_steps):
    """Extract stable rank from pickle data."""
    if data is None or 'stable_rank' not in data:
        return None, None

    srank = data['stable_rank']
    if isinstance(srank, torch.Tensor):
        srank = srank.numpy()

    n_points = len(srank)
    # Assume evenly spaced logging
    steps = np.linspace(0, total_steps, n_points)
    return steps, srank


def extract_weight_magnitude(data, total_steps):
    """Extract weight magnitude from pickle data."""
    if data is None or 'pol_weights' not in data:
        return None, None

    weights = data['pol_weights']
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()

    # pol_weights shape is (n_logs, 3) where columns might be [mean, std, norm] or similar
    # Let's take the first column as the main weight metric
    n_points = weights.shape[0]
    steps = np.linspace(0, total_steps, n_points)

    # Use the mean (first column) or compute L2 norm
    weight_mag = weights[:, 0]  # Adjust based on what's stored
    return steps, weight_mag


def extract_margin_dead_pct(data, total_steps, margin_type='actual'):
    """Extract dead percentage from margin data."""
    if data is None or 'margin_data' not in data:
        return None, None

    margin_data = data['margin_data']
    if margin_type not in margin_data:
        return None, None

    margins = margin_data[margin_type]
    timestamps = np.array(margins['timestamps'])

    # dead_counts is a list of dicts with layer indices as keys
    # Sum across all layers and compute percentage
    dead_counts_list = margins['dead_counts']
    dead_pct = []

    # Total neurons: 256 + 256 = 512 (2 hidden layers of 256 each)
    total_neurons = 512

    for d in dead_counts_list:
        if isinstance(d, dict):
            total_dead = sum(d.values())
            dead_pct.append(total_dead / total_neurons * 100)
        else:
            dead_pct.append(d / total_neurons * 100)

    return timestamps, np.array(dead_pct)


def plot_margin_percent(env, margin_type='actual'):
    """Plot margin dead percentage for an environment."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    total_steps = TOTAL_STEPS[env]
    max_val = 0

    for method in ['std', 'cbp', 'ns', 'l2']:
        filepath = PICKLE_LOGS[env][method]
        data = load_pickle_log(filepath)
        steps, values = extract_margin_dead_pct(data, total_steps, margin_type)

        if steps is not None and len(steps) > 0:
            steps_millions = steps / 1e6
            ax.plot(steps_millions, values, color=COLORS[method], label=LABELS[method], linewidth=1.5)
            max_val = max(max_val, np.nanmax(values))

    # Add friction change lines for SlipperyAnt
    if env == 'sant':
        for fc in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel('Dead Neurons (%)')
    env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
    title_name = 'Predictive Margin' if margin_type == 'predictive' else 'Actual Margin'
    ax.set_title(f'{env_title} - {title_name} Dead %')
    ax.legend(loc='best')

    xlim = 100 if env == 'ant' else 20
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 100)

    plt.tight_layout()
    plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/margin_{margin_type}_{env}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: margin_{margin_type}_{env}.png")


def plot_metric(env, metric_name, ylabel, title_suffix, data_func, output_name, ylim_zero=True, smooth=False):
    """Generic function to plot a metric with optional smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    max_val = 0
    for method in ['std', 'cbp', 'ns', 'l2']:
        filepath = OUTPUT_LOGS[env][method]
        steps, values = data_func(filepath)
        if len(steps) > 0:
            steps_millions = steps / 1e6
            if ylim_zero:
                values = np.maximum(values, 0)

            if smooth:
                # Compute rolling mean and std
                mean, std = rolling_mean_std(values)
                ax.plot(steps_millions, mean, color=COLORS[method], label=LABELS[method], linewidth=1.5)
                ax.fill_between(steps_millions, mean - std, mean + std,
                               color=COLORS[method], alpha=0.2)
                max_val = max(max_val, np.nanmax(mean + std))
            else:
                # Plot raw data
                ax.plot(steps_millions, values, color=COLORS[method], label=LABELS[method], linewidth=1.5)
                max_val = max(max_val, np.nanmax(values))

    # Add friction change lines for SlipperyAnt
    if env == 'sant':
        for fc in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel(ylabel)
    env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
    ax.set_title(f'{env_title} - {title_suffix}')
    ax.legend(loc='best')

    xlim = 100 if env == 'ant' else 20
    ax.set_xlim(0, xlim)
    if ylim_zero and max_val > 0:
        ax.set_ylim(0, max_val * 1.1)

    plt.tight_layout()
    plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/{output_name}_{env}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}_{env}.png")


def plot_dormant_vs_margin(env, margin_type='actual'):
    """Plot comparison of dormant units vs margin dead% for each method."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    total_steps = TOTAL_STEPS[env]

    for idx, method in enumerate(['std', 'cbp', 'ns', 'l2']):
        ax = axes[idx]

        # Get dormant units from output logs
        dormant_steps, dormant_pct = parse_output_dormant(OUTPUT_LOGS[env][method])

        # Get margin dead% from pickle logs
        data = load_pickle_log(PICKLE_LOGS[env][method])
        margin_steps, margin_pct = extract_margin_dead_pct(data, total_steps, margin_type)

        # Plot margin dead% with HEAVY SMOOTHING (window=50) to reduce noise
        if margin_steps is not None and len(margin_steps) > 0:
            margin_steps_m = margin_steps / 1e6
            # Apply heavy smoothing to margin (window=50)
            margin_smoothed, margin_std = rolling_mean_std(margin_pct, window=50)
            ax.plot(margin_steps_m, margin_smoothed, color='red',
                   label=f'{margin_type.capitalize()} Margin (smoothed)',
                   linewidth=2.5)
            # Add light shaded area for variance
            ax.fill_between(margin_steps_m,
                           np.maximum(margin_smoothed - margin_std, 0),
                           margin_smoothed + margin_std,
                           color='red', alpha=0.15)

        # Plot dormant units with HEAVY SMOOTHING too
        if len(dormant_steps) > 0:
            dormant_steps_m = dormant_steps / 1e6
            # Apply heavy smoothing to dormant (window=30)
            dormant_smoothed, dormant_std = rolling_mean_std(dormant_pct, window=30)
            ax.plot(dormant_steps_m, dormant_smoothed, color='blue', label='Dormant Units (smoothed)',
                   linewidth=2.5)
            # Add light shaded area for variance
            ax.fill_between(dormant_steps_m,
                           np.maximum(dormant_smoothed - dormant_std, 0),
                           dormant_smoothed + dormant_std,
                           color='blue', alpha=0.15)

        # Add friction change lines for SlipperyAnt
        if env == 'sant':
            for fc in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
                ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel('Steps (millions)')
        ax.set_ylabel('Dead/Dormant (%)')
        ax.set_title(f'{LABELS[method]}')
        ax.legend(loc='best')

        xlim = 100 if env == 'ant' else 20
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, None)

    env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
    fig.suptitle(f'{env_title} - Dormant Units vs {margin_type.capitalize()} Margin Dead%', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/dormant_vs_margin_{margin_type}_{env}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: dormant_vs_margin_{margin_type}_{env}.png")


def plot_pickle_metric(env, metric_name, ylabel, title_suffix, extract_func, output_name, ylim_zero=True):
    """Plot metrics from pickle files with shaded variance."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    max_val = 0
    total_steps = TOTAL_STEPS[env]

    for method in ['std', 'cbp', 'ns', 'l2']:
        filepath = PICKLE_LOGS[env][method]
        data = load_pickle_log(filepath)
        steps, values = extract_func(data, total_steps)

        if steps is not None and len(steps) > 0:
            steps_millions = steps / 1e6
            if ylim_zero:
                values = np.maximum(values, 0)

            # Compute rolling mean and std
            mean, std = rolling_mean_std(values)

            # Plot mean line
            ax.plot(steps_millions, mean, color=COLORS[method], label=LABELS[method], linewidth=1.5)

            # Add shaded area for variance
            ax.fill_between(steps_millions, mean - std, mean + std,
                           color=COLORS[method], alpha=0.2)

            max_val = max(max_val, np.nanmax(mean + std))

    # Add friction change lines for SlipperyAnt
    if env == 'sant':
        for fc in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel(ylabel)
    env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
    ax.set_title(f'{env_title} - {title_suffix}')
    ax.legend(loc='best')

    xlim = 100 if env == 'ant' else 20
    ax.set_xlim(0, xlim)
    if ylim_zero and max_val > 0:
        ax.set_ylim(0, max_val * 1.1)

    plt.tight_layout()
    plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/{output_name}_{env}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}_{env}.png")


if __name__ == '__main__':
    print("=" * 50)
    print("Plotting from output logs...")
    print("=" * 50)

    # Returns (raw data, no smoothing)
    print("\nPlotting returns...")
    for env in ['ant', 'sant']:
        plot_metric(env, 'returns', 'Average Return', 'Average Return',
                   parse_output_returns, 'returns', ylim_zero=True, smooth=False)

    # Dormant units (raw data, no smoothing)
    print("\nPlotting dormant units...")
    for env in ['ant', 'sant']:
        plot_metric(env, 'dormant', 'Dormant Units (%)', 'Dormant Units',
                   parse_output_dormant, 'dormant', ylim_zero=True, smooth=False)

    print("\n" + "=" * 50)
    print("Plotting from pickle logs...")
    print("=" * 50)

    # Stable rank
    print("\nPlotting stable rank...")
    for env in ['ant', 'sant']:
        plot_pickle_metric(env, 'stable_rank', 'Stable Rank', 'Stable Rank',
                          extract_stable_rank, 'stable_rank', ylim_zero=True)

    # Weight magnitude
    print("\nPlotting weight magnitude...")
    for env in ['ant', 'sant']:
        plot_pickle_metric(env, 'weight_mag', 'Weight Magnitude', 'Weight Magnitude',
                          extract_weight_magnitude, 'weight_magnitude', ylim_zero=True)

    # Margin - predictive and actual
    print("\nPlotting margin (predictive)...")
    for env in ['ant', 'sant']:
        plot_margin_percent(env, margin_type='predictive')

    print("\nPlotting margin (actual)...")
    for env in ['ant', 'sant']:
        plot_margin_percent(env, margin_type='actual')

    # Comparison: Dormant vs Margin Dead%
    print("\nPlotting dormant vs margin comparison...")
    for env in ['ant', 'sant']:
        plot_dormant_vs_margin(env, margin_type='actual')
        plot_dormant_vs_margin(env, margin_type='predictive')
