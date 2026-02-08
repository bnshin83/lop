#!/usr/bin/env python3
"""
Windowed plasticity loss analysis: Compare 10-epoch averages before vs after each boundary
to filter out immediate transition noise.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_sample_memo_map():
    """Load memorization scores mapping."""
    csv_file = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Sample mapping file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    sample_map = {}
    for _, row in df.iterrows():
        sample_map[int(row['sample_id'])] = float(row['memorization_score'])
    
    print(f"Loaded memorization scores for {len(sample_map)} samples")
    return sample_map

def get_task_boundaries():
    """Define task boundaries (every 200 epochs)."""
    boundaries = []
    for i in range(200, 4001, 200):
        boundaries.append(i)
    
    print(f"Task boundaries: {boundaries[:5]}...{boundaries[-5:]} ({len(boundaries)} total)")
    return boundaries

def load_epoch_data(losses_dir, dormant_dir, epoch):
    """Load loss and dormant data for a specific epoch."""
    
    epoch_str = f"epoch_{epoch:04d}"
    
    # Load loss data
    loss_file = os.path.join(losses_dir, epoch_str, "combined_loss_data.csv")
    loss_data = None
    if os.path.exists(loss_file):
        try:
            loss_data = pd.read_csv(loss_file)
        except Exception as e:
            print(f"Warning: Could not load loss data for epoch {epoch}: {e}")
    
    # Load dormant data
    dormant_file = os.path.join(dormant_dir, epoch_str, "combined_dormant_data.csv")
    dormant_data = None
    if os.path.exists(dormant_file):
        try:
            dormant_data = pd.read_csv(dormant_file)
        except Exception as e:
            print(f"Warning: Could not load dormant data for epoch {epoch}: {e}")
    
    return loss_data, dormant_data

def compute_windowed_averages(losses_dir, dormant_dir, epochs):
    """
    Compute averaged metrics over a window of epochs.
    
    Args:
        losses_dir: Path to losses directory
        dormant_dir: Path to dormant directory  
        epochs: List of epochs to average over
    
    Returns:
        DataFrame with averaged metrics per sample
    """
    
    all_data = []
    
    for epoch in epochs:
        loss_data, dormant_data = load_epoch_data(losses_dir, dormant_dir, epoch)
        
        if loss_data is not None and dormant_data is not None:
            # Merge loss and dormant data
            merged = pd.merge(loss_data, dormant_data, on=['sample_id', 'class_label'], 
                            suffixes=('', '_dorm'))
            merged['epoch'] = epoch
            all_data.append(merged)
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all epochs
    combined = pd.concat(all_data, ignore_index=True)
    
    # Compute averages per sample
    averaged = combined.groupby(['sample_id', 'class_label']).agg({
        'loss': 'mean',
        'loss_grad': 'mean', 
        'dormant_prop': 'mean'
    }).reset_index()
    
    return averaged

def compute_windowed_plasticity_loss(losses_dir, dormant_dir, memo_map, boundary_epoch, window_size=10):
    """
    Compute plasticity loss using windowed averages to reduce noise.
    
    Args:
        losses_dir: Path to losses directory
        dormant_dir: Path to dormant directory
        memo_map: Memorization scores mapping
        boundary_epoch: The boundary epoch (e.g., 400)
        window_size: Number of epochs to average (default 10)
        
    Returns:
        DataFrame with windowed plasticity loss metrics
    """
    
    print(f"Computing windowed plasticity loss at boundary: epoch {boundary_epoch}")
    print(f"  Pre-boundary window: epochs {boundary_epoch-window_size+1} to {boundary_epoch}")
    print(f"  Post-boundary window: epochs {boundary_epoch+1} to {boundary_epoch+window_size}")
    
    # Define epoch windows
    pre_epochs = list(range(boundary_epoch - window_size + 1, boundary_epoch + 1))
    post_epochs = list(range(boundary_epoch + 1, boundary_epoch + window_size + 1))
    
    # Check if we have enough epochs (avoid going below epoch 1 or above 4000)
    pre_epochs = [e for e in pre_epochs if e >= 1]
    post_epochs = [e for e in post_epochs if e <= 4000]
    
    if len(pre_epochs) < window_size // 2 or len(post_epochs) < window_size // 2:
        print(f"Warning: Insufficient epochs for boundary {boundary_epoch}")
        return pd.DataFrame()
    
    # Compute windowed averages
    pre_data = compute_windowed_averages(losses_dir, dormant_dir, pre_epochs)
    post_data = compute_windowed_averages(losses_dir, dormant_dir, post_epochs)
    
    if pre_data.empty or post_data.empty:
        print(f"Warning: No data for windowed analysis at boundary {boundary_epoch}")
        return pd.DataFrame()
    
    # Merge pre and post data
    combined_data = pd.merge(pre_data, post_data, on=['sample_id', 'class_label'], 
                            suffixes=('_pre', '_post'))
    
    if combined_data.empty:
        print(f"Warning: No overlapping samples for windowed boundary {boundary_epoch}")
        return pd.DataFrame()
    
    # Add memorization scores
    combined_data['memo_score'] = combined_data['sample_id'].map(memo_map)
    combined_data = combined_data.dropna(subset=['memo_score'])
    
    if combined_data.empty:
        print(f"Warning: No samples with memo scores for windowed boundary {boundary_epoch}")
        return pd.DataFrame()
    
    # Compute windowed plasticity loss metrics
    combined_data['delta_loss'] = combined_data['loss_post'] - combined_data['loss_pre']
    combined_data['delta_grad_norm'] = combined_data['loss_grad_pre'] - combined_data['loss_grad_post']  # Grad norm drop
    combined_data['delta_dormant'] = combined_data['dormant_prop_post'] - combined_data['dormant_prop_pre']
    
    # Add metadata
    combined_data['boundary_epoch'] = boundary_epoch
    combined_data['task_number'] = (boundary_epoch - 1) // 200
    combined_data['window_size'] = window_size
    combined_data['n_pre_epochs'] = len(pre_epochs)
    combined_data['n_post_epochs'] = len(post_epochs)
    
    print(f"  Processed {len(combined_data)} samples with {len(pre_epochs)} pre + {len(post_epochs)} post epochs")
    
    return combined_data

def compute_all_correlations(x, y):
    """Compute all correlation types between x and y."""
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    if len(x_clean) < 10:
        return {key: np.nan for key in ['pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'kendall_r', 'kendall_p']}
    
    results = {}
    
    # Pearson (linear)
    try:
        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
        results['pearson_r'] = pearson_r
        results['pearson_p'] = pearson_p
    except:
        results['pearson_r'] = np.nan
        results['pearson_p'] = np.nan
    
    # Spearman (monotonic)
    try:
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
        results['spearman_r'] = spearman_r
        results['spearman_p'] = spearman_p
    except:
        results['spearman_r'] = np.nan
        results['spearman_p'] = np.nan
    
    # Kendall (rank-based)
    try:
        kendall_r, kendall_p = stats.kendalltau(x_clean, y_clean)
        results['kendall_r'] = kendall_r
        results['kendall_p'] = kendall_p
    except:
        results['kendall_r'] = np.nan
        results['kendall_p'] = np.nan
    
    return results

def analyze_windowed_correlations(windowed_data):
    """Analyze correlations using windowed plasticity loss data."""
    
    print("\n" + "="*80)
    print("WINDOWED CORRELATION ANALYSIS (10-epoch windows)")
    print("="*80)
    
    results = []
    
    plasticity_probes = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    probe_labels = ['Δloss', 'Δgrad_norm', 'Δdormant']
    
    for boundary_epoch, group in windowed_data.groupby('boundary_epoch'):
        if len(group) < 20:
            continue
        
        boundary_results = {
            'boundary_epoch': boundary_epoch,
            'task_number': group['task_number'].iloc[0],
            'n_samples': len(group),
            'window_size': group['window_size'].iloc[0],
            'n_pre_epochs': group['n_pre_epochs'].iloc[0],
            'n_post_epochs': group['n_post_epochs'].iloc[0]
        }
        
        # Compute correlations for each plasticity probe
        for probe, label in zip(plasticity_probes, probe_labels):
            if probe in group.columns:
                correlations = compute_all_correlations(group['memo_score'], group[probe])
                
                for corr_type, value in correlations.items():
                    boundary_results[f'memo_{probe}_{corr_type}'] = value
        
        # Add means for context
        boundary_results['mean_memo'] = group['memo_score'].mean()
        boundary_results['mean_delta_loss'] = group['delta_loss'].mean()
        boundary_results['mean_delta_grad_norm'] = group['delta_grad_norm'].mean()
        boundary_results['mean_delta_dormant'] = group['delta_dormant'].mean()
        
        results.append(boundary_results)
    
    results_df = pd.DataFrame(results).sort_values('task_number')
    
    print(f"Analyzed windowed correlations for {len(results_df)} boundaries")
    
    return results_df

def print_windowed_correlation_summary(results_df):
    """Print summary of windowed correlation analysis."""
    
    plasticity_probes = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    probe_labels = ['Δloss (windowed)', 'Δgrad_norm (windowed)', 'Δdormant (windowed)']
    correlation_types = ['pearson_r', 'spearman_r', 'kendall_r']
    
    print(f"\nOverall windowed correlation strengths (mean ± std across {len(results_df)} boundaries):")
    
    for probe, label in zip(plasticity_probes, probe_labels):
        print(f"\nMemo ↔ {label}:")
        for corr_type in correlation_types:
            col = f'memo_{probe}_{corr_type}'
            if col in results_df.columns:
                values = results_df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    print(f"  {corr_type.replace('_r', '').capitalize():>8}: {mean_val:7.4f} ± {std_val:.4f}")
    
    print(f"\n" + "="*60)
    print("WINDOWED EVOLUTION TRENDS")
    print("="*60)
    
    for probe, label in zip(plasticity_probes, probe_labels):
        print(f"\nMemo ↔ {label} windowed evolution:")
        
        for corr_type in correlation_types:
            col = f'memo_{probe}_{corr_type}'
            if col in results_df.columns:
                values = results_df[col].dropna()
                if len(values) > 1:
                    task_numbers = results_df['task_number'][:len(values)]
                    trend_corr, trend_p = stats.spearmanr(task_numbers, values)
                    
                    if trend_p < 0.05:
                        if abs(trend_corr) > 0.5:
                            trend_strength = "STRONG"
                        elif abs(trend_corr) > 0.3:
                            trend_strength = "MODERATE"
                        else:
                            trend_strength = "WEAK"
                            
                        direction = "INCREASING" if trend_corr > 0 else "DECREASING"
                        significance = "***" if trend_p < 0.001 else "**" if trend_p < 0.01 else "*"
                        
                        print(f"  {corr_type.replace('_r', '').capitalize():>8}: {trend_strength} {direction} {significance} "
                              f"(trend ρ={trend_corr:.3f}, p={trend_p:.4f})")
                        
                        min_val, max_val = values.min(), values.max()
                        print(f"           Range: {min_val:.4f} → {max_val:.4f} (Δ={max_val-min_val:+.4f})")
                    else:
                        print(f"  {corr_type.replace('_r', '').capitalize():>8}: No significant trend (p={trend_p:.3f})")
    
    print(f"\n" + "="*60)
    print("WINDOWED TASK-BY-TASK BREAKDOWN")
    print("="*60)
    
    print(f"Windowed Memo ↔ Δdormant correlations (10-epoch averages):")
    print(f"{'Task':>4} {'Epoch':>5} {'Pearson':>8} {'Spearman':>9} {'Kendall':>8} {'N':>6} {'Pre':>4} {'Post':>4}")
    print("-" * 65)
    
    for _, row in results_df.iterrows():
        task = int(row['task_number'])
        epoch = int(row['boundary_epoch'])
        n_samples = int(row['n_samples'])
        n_pre = int(row['n_pre_epochs'])
        n_post = int(row['n_post_epochs'])
        
        pearson = row.get('memo_delta_dormant_pearson_r', np.nan)
        spearman = row.get('memo_delta_dormant_spearman_r', np.nan)
        kendall = row.get('memo_delta_dormant_kendall_r', np.nan)
        
        print(f"{task:>4} {epoch:>5} {pearson:>8.4f} {spearman:>9.4f} {kendall:>8.4f} {n_samples:>6} {n_pre:>4} {n_post:>4}")

def create_windowed_correlation_plots(results_df, windowed_data):
    """Create plots comparing windowed vs immediate correlations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    plasticity_probes = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    probe_labels = ['Memo ↔ Δloss (Windowed)', 'Memo ↔ Δgrad_norm (Windowed)', 'Memo ↔ Δdormant (Windowed)']
    colors = ['red', 'blue', 'green']
    
    # Plot 1-3: Windowed correlation evolution
    correlation_types = ['pearson_r', 'spearman_r', 'kendall_r']
    type_labels = ['Pearson', 'Spearman', 'Kendall']
    type_colors = ['blue', 'red', 'green']
    
    for i, (probe, probe_label, color) in enumerate(zip(plasticity_probes, probe_labels, colors)):
        ax = axes[0, i]
        
        for corr_type, type_label, type_color in zip(correlation_types, type_labels, type_colors):
            col = f'memo_{probe}_{corr_type}'
            if col in results_df.columns:
                values = results_df[col].dropna()
                tasks = results_df['task_number'][:len(values)]
                
                ax.plot(tasks, values, 'o-', color=type_color, linewidth=2, 
                       markersize=6, label=f'{type_label} (windowed)', alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Task Number')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'{probe_label}', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Δdormant trend analysis
    ax4 = axes[1, 0]
    spearman_col = 'memo_delta_dormant_spearman_r'
    if spearman_col in results_df.columns:
        values = results_df[spearman_col].dropna()
        tasks = results_df['task_number'][:len(values)]
        
        ax4.scatter(tasks, values, s=100, alpha=0.7, color='green', zorder=3, label='Windowed')
        
        if len(values) > 1:
            z = np.polyfit(tasks, values, 1)
            p = np.poly1d(z)
            ax4.plot(tasks, p(tasks), "b--", linewidth=3, alpha=0.8)
            
            trend_corr, trend_p = stats.spearmanr(tasks, values)
            significance = "***" if trend_p < 0.001 else "**" if trend_p < 0.01 else "*" if trend_p < 0.05 else ""
            
            ax4.text(0.05, 0.95, f'Windowed Trend: ρ={trend_corr:.3f} {significance}\np={trend_p:.4f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
                    verticalalignment='top')
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Task Number')
        ax4.set_ylabel('Spearman Correlation')
        ax4.set_title('Windowed Memo ↔ Δdormant Trend', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Compare all windowed probes
    ax5 = axes[1, 1]
    for probe, probe_label, color in zip(plasticity_probes, probe_labels, colors):
        col = f'memo_{probe}_spearman_r'
        if col in results_df.columns:
            values = results_df[col].dropna()
            tasks = results_df['task_number'][:len(values)]
            clean_label = probe_label.split(' ↔ ')[1].replace(' (Windowed)', '')
            ax5.plot(tasks, values, 'o-', color=color, linewidth=2.5, 
                    markersize=7, label=clean_label, alpha=0.8)
    
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Task Number')
    ax5.set_ylabel('Spearman Correlation (Windowed)')
    ax5.set_title('All Windowed Probes Comparison', fontweight='bold', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Sample statistics
    ax6 = axes[1, 2]
    ax6.bar(results_df['task_number'], results_df['n_samples'], alpha=0.7, color='orange')
    ax6.set_xlabel('Task Number')
    ax6.set_ylabel('Number of Samples')
    ax6.set_title('Sample Count per Boundary (Windowed)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add text showing window info
    if not results_df.empty:
        window_size = results_df['window_size'].iloc[0]
        fig.suptitle(f'Windowed Plasticity Analysis ({window_size}-epoch windows)', 
                     fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    return fig

def main():
    """Main windowed analysis function."""
    
    print("WINDOWED PLASTICITY LOSS ANALYSIS")
    print("="*60)
    print("Using 10-epoch windows to reduce boundary transition noise")
    
    # Paths
    losses_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_losses_full"
    dormant_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/per_sample_dormant_analysis"
    
    # Load memorization scores
    memo_map = load_sample_memo_map()
    if memo_map is None:
        return
    
    # Get task boundaries
    boundaries = get_task_boundaries()
    
    # Collect windowed plasticity loss data
    all_windowed_data = []
    window_size = 10
    
    print(f"\nProcessing {len(boundaries)} task boundaries with {window_size}-epoch windows...")
    
    for i, boundary_epoch in enumerate(boundaries):
        if i == 0:  # Skip first boundary (not enough pre-epochs)
            continue
            
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(boundaries)} boundaries...")
        
        windowed_data = compute_windowed_plasticity_loss(
            losses_dir, dormant_dir, memo_map, boundary_epoch, window_size)
        
        if not windowed_data.empty:
            all_windowed_data.append(windowed_data)
    
    if not all_windowed_data:
        print("Error: No windowed data collected!")
        return
    
    # Combine all windowed data
    combined_windowed_data = pd.concat(all_windowed_data, ignore_index=True)
    print(f"\nTotal samples across all windowed boundaries: {len(combined_windowed_data)}")
    
    # Analyze windowed correlations
    windowed_results = analyze_windowed_correlations(combined_windowed_data)
    
    # Print summary
    print_windowed_correlation_summary(windowed_results)
    
    # Save results
    output_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system"
    
    # Save windowed data
    windowed_data_file = os.path.join(output_dir, "windowed_plasticity_data.csv")
    combined_windowed_data.to_csv(windowed_data_file, index=False)
    print(f"\nWindowed data saved to: {windowed_data_file}")
    
    # Save windowed results
    windowed_results_file = os.path.join(output_dir, "windowed_correlation_results.csv")
    windowed_results.to_csv(windowed_results_file, index=False)
    print(f"Windowed correlation results saved to: {windowed_results_file}")
    
    # Create plots
    print("\nCreating windowed correlation plots...")
    fig = create_windowed_correlation_plots(windowed_results, combined_windowed_data)
    
    plot_file = os.path.join(output_dir, "windowed_correlation_plots.png")
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Windowed plots saved to: {plot_file}")
    
    print("\nWindowed plasticity loss analysis complete!")

if __name__ == "__main__":
    main()