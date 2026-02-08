#!/usr/bin/env python3
"""Plot return curves for Ant and SlipperyAnt experiments."""

import re
import matplotlib.pyplot as plt
import numpy as np

# Style settings to match reference plot
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

def parse_log_file(filepath):
    """Parse a log file and extract step and return data."""
    steps = []
    returns = []

    with open(filepath, 'r') as f:
        for line in f:
            # Match lines like: Step 1000000/100000000 (1.0%) | Episodes: 8979 | Avg Return (last 100): 750.24
            match = re.search(r'Step (\d+)/\d+.*Avg Return \(last 100\): ([-\d.]+)', line)
            if match:
                step = int(match.group(1))
                ret = float(match.group(2))
                steps.append(step)
                returns.append(ret)

    return np.array(steps), np.array(returns)


def plot_ant_returns():
    """Create return plot for Ant."""

    base_path = '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs'

    ant_experiments = {
        'std': f'{base_path}/5228737_ant_std.out',
        'cbp': f'{base_path}/5228714_ant_cbp.out',
        'ns': f'{base_path}/5228719_ant_ns.out',
        'l2': f'{base_path}/5228718_ant_l2.out',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    max_return = 0
    for method, filepath in ant_experiments.items():
        steps, returns = parse_log_file(filepath)
        steps_millions = steps / 1e6
        # Clip negative returns to 0 for visualization
        returns_clipped = np.maximum(returns, 0)
        ax.plot(steps_millions, returns_clipped, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        max_return = max(max_return, np.max(returns_clipped))

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel('Average Return')
    ax.set_title('Ant-v3')
    ax.legend(loc='best')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_return * 1.1)

    plt.tight_layout()
    plt.savefig('/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/returns_ant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: returns_ant.png")


def plot_sant_returns():
    """Create return plot for SlipperyAnt."""

    base_path = '/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/logs'

    sant_experiments = {
        'std': f'{base_path}/5228738_sant_std.out',
        'cbp': f'{base_path}/5228724_sant_cbp.out',
        'ns': f'{base_path}/5228721_sant_ns.out',
        'l2': f'{base_path}/5228722_sant_l2.out',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    max_return = 0
    for method, filepath in sant_experiments.items():
        steps, returns = parse_log_file(filepath)
        steps_millions = steps / 1e6
        # Clip negative returns to 0 for visualization
        returns_clipped = np.maximum(returns, 0)
        ax.plot(steps_millions, returns_clipped, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        max_return = max(max_return, np.max(returns_clipped))

    # Add vertical lines for friction changes in SlipperyAnt
    friction_changes = [2, 4, 6, 8, 10, 12, 14, 16, 18]  # Every 2M steps
    for fc in friction_changes:
        ax.axvline(x=fc, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Steps (millions)')
    ax.set_ylabel('Average Return')
    ax.set_title('SlipperyAnt-v3 (dashed lines = friction changes)')
    ax.legend(loc='best')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, max_return * 1.1)

    plt.tight_layout()
    plt.savefig('/scratch/gautschi/shin283/loss-of-plasticity/lop/rl/returns_sant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: returns_sant.png")


if __name__ == '__main__':
    plot_ant_returns()
    plot_sant_returns()
