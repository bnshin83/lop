#!/usr/bin/env python3
"""Plot all metrics for Ant and SlipperyAnt experiments."""

import re
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Style settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

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

# WandB run IDs (extracted from config files)
WANDB_RUNS = {
    'ant': {
        'cbp': 'etrd60g0',
        'l2': 'y84hl7sw',
        'ns': 'pdm9xdw1',
        'std': 'y4anycah',
    },
    'sant': {
        'cbp': 'etrd60g0',
        'l2': 'y84hl7sw',
        'ns': 'pdm9xdw1',
        'std': 'y4anycah',
    }
}

# Log file paths
LOG_FILES = {
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

def parse_log_returns(filepath):
    """Parse a log file and extract step and return data."""
    steps = []
    returns = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'Step (\d+)/\d+.*Avg Return \(last 100\): ([-\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                returns.append(float(match.group(2)))
    return np.array(steps), np.array(returns)


def parse_log_dormant(filepath):
    """Parse a log file and extract step and dormant unit percentage."""
    steps = []
    dormant_pct = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\[Update \d+\] Step (\d+): Actual Dead=\d+/\d+ \(([\d.]+)%\)', line)
            if match:
                steps.append(int(match.group(1)))
                dormant_pct.append(float(match.group(2)))
    return np.array(steps), np.array(dormant_pct)


def fetch_wandb_data(entity, project, run_id, metric_keys):
    """Fetch data from WandB for given metrics."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.scan_history(keys=metric_keys + ['global_step'])

    data = {key: [] for key in metric_keys}
    data['steps'] = []

    for row in history:
        if 'global_step' in row:
            data['steps'].append(row['global_step'])
            for key in metric_keys:
                data[key].append(row.get(key, np.nan))

    return {k: np.array(v) for k, v in data.items()}


def plot_single_metric(env, metric_name, ylabel, title_suffix, get_data_func, output_name, ylim_zero=True):
    """Generic function to plot a single metric for one environment."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    max_val = 0
    for method, filepath in LOG_FILES[env].items():
        steps, values = get_data_func(filepath)
        steps_millions = steps / 1e6
        if ylim_zero:
            values = np.maximum(values, 0)
        ax.plot(steps_millions, values, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        if len(values) > 0:
            max_val = max(max_val, np.nanmax(values))

    # Add friction change lines for SlipperyAnt
    if env == 'sant':
        friction_changes = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        for fc in friction_changes:
            ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel(ylabel)
    env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
    ax.set_title(f'{env_title} - {title_suffix}')
    ax.legend(loc='best')

    if env == 'ant':
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(0, 20)

    if ylim_zero and max_val > 0:
        ax.set_ylim(0, max_val * 1.1)

    plt.tight_layout()
    plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/{output_name}_{env}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}_{env}.png")


def plot_returns():
    """Plot returns for both environments."""
    for env in ['ant', 'sant']:
        plot_single_metric(env, 'returns', 'Average Return', 'Returns',
                          parse_log_returns, 'returns', ylim_zero=True)


def plot_dormant():
    """Plot dormant units for both environments."""
    for env in ['ant', 'sant']:
        plot_single_metric(env, 'dormant', 'Dormant Units (%)', 'Dormant Units',
                          parse_log_dormant, 'dormant', ylim_zero=True)


def plot_wandb_metrics():
    """Plot stable rank and weight magnitude from WandB."""
    api = wandb.Api()
    entity = "shin283"  # Update if different
    project = "lop-rl"  # Update if different

    # Metrics to fetch
    metrics = {
        'stable_rank': ('pol_srank', 'Stable Rank', 'Stable Rank'),
        'weight_mag': ('pol_weight_mag', 'Weight Magnitude', 'Weight Magnitude'),
    }

    for env in ['ant', 'sant']:
        for metric_key, (wandb_key, ylabel, title) in metrics.items():
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            for method, run_id in WANDB_RUNS[env].items():
                try:
                    run = api.run(f"{entity}/{project}/{run_id}")
                    history = list(run.scan_history(keys=[wandb_key, 'global_step']))

                    steps = [r['global_step'] for r in history if wandb_key in r and 'global_step' in r]
                    values = [r[wandb_key] for r in history if wandb_key in r and 'global_step' in r]

                    if steps:
                        steps_millions = np.array(steps) / 1e6
                        ax.plot(steps_millions, values, color=COLORS[method],
                               label=LABELS[method], linewidth=1.5)
                except Exception as e:
                    print(f"Error fetching {wandb_key} for {env}/{method}: {e}")

            # Add friction change lines for SlipperyAnt
            if env == 'sant':
                friction_changes = [2, 4, 6, 8, 10, 12, 14, 16, 18]
                for fc in friction_changes:
                    ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

            ax.set_xlabel('Steps (millions)')
            ax.set_ylabel(ylabel)
            env_title = 'Ant-v3' if env == 'ant' else 'SlipperyAnt-v3'
            ax.set_title(f'{env_title} - {title}')
            ax.legend(loc='best')

            if env == 'ant':
                ax.set_xlim(0, 100)
            else:
                ax.set_xlim(0, 20)

            ax.set_ylim(0, None)

            plt.tight_layout()
            plt.savefig(f'/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/{metric_key}_{env}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {metric_key}_{env}.png")


if __name__ == '__main__':
    # Plot from log files (no WandB needed)
    print("Plotting returns...")
    plot_returns()

    print("\nPlotting dormant units...")
    plot_dormant()

    # Uncomment to fetch from WandB (requires authentication)
    # print("\nPlotting WandB metrics...")
    # plot_wandb_metrics()
