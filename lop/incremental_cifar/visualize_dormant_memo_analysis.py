#!/usr/bin/env python3
"""
Visualize the relationship between dormant unit proportions and memorization scores.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def load_analysis_data():
    """Load the dormant-memo analysis results."""
    data_file = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/top_dormant_memo_analysis.csv"
    
    if not Path(data_file).exists():
        print(f"Error: Analysis file not found: {data_file}")
        print("Please run analyze_top_dormant_memo_scores.py first.")
        return None
    
    df = pd.read_csv(data_file)
    print(f"Loaded data for {len(df)} epochs")
    return df

def create_comprehensive_visualization(df):
    """Create comprehensive visualization of dormant-memo relationship."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Time series of memorization scores
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(df['epoch'], df['avg_memo_score'], linewidth=1.5, alpha=0.8, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Memorization Score')
    plt.title('Memorization Score of Top Dormant Sample Over Training', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. Time series of dormant proportions
    ax2 = plt.subplot(3, 2, 2)
    plt.plot(df['epoch'], df['top_dormant_prop'], linewidth=1.5, alpha=0.8, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Dormant Unit Proportion')
    plt.title('Dormant Proportion of Top Sample Over Training', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Dormant vs Memo score
    ax3 = plt.subplot(3, 2, 3)
    scatter = plt.scatter(df['top_dormant_prop'], df['avg_memo_score'], 
                         c=df['epoch'], cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Dormant Unit Proportion')
    plt.ylabel('Memorization Score')
    plt.title('Dormant Proportion vs Memorization Score\n(Color = Epoch)', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Epoch')
    
    # Add correlation coefficient
    corr_coeff = df['top_dormant_prop'].corr(df['avg_memo_score'])
    plt.text(0.05, 0.95, f'Correlation: {corr_coeff:.3f}', 
             transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    # 4. Histogram of memorization scores
    ax4 = plt.subplot(3, 2, 4)
    plt.hist(df['avg_memo_score'], bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Memorization Score')
    plt.ylabel('Frequency (Epochs)')
    plt.title('Distribution of Memorization Scores', fontweight='bold', fontsize=12)
    plt.axvline(df['avg_memo_score'].mean(), color='darkred', linestyle='--', 
                label=f'Mean: {df["avg_memo_score"].mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Histogram of dormant proportions
    ax5 = plt.subplot(3, 2, 5)
    plt.hist(df['top_dormant_prop'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Dormant Unit Proportion')
    plt.ylabel('Frequency (Epochs)')
    plt.title('Distribution of Dormant Proportions', fontweight='bold', fontsize=12)
    plt.axvline(df['top_dormant_prop'].mean(), color='darkblue', linestyle='--', 
                label=f'Mean: {df["top_dormant_prop"].mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Rolling averages
    ax6 = plt.subplot(3, 2, 6)
    window = 100  # 100-epoch rolling window
    df_sorted = df.sort_values('epoch')
    memo_rolling = df_sorted['avg_memo_score'].rolling(window=window, center=True).mean()
    dormant_rolling = df_sorted['top_dormant_prop'].rolling(window=window, center=True).mean()
    
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(df_sorted['epoch'], memo_rolling, 'r-', linewidth=2, 
                     label=f'Memo Score (rolling avg, window={window})')
    line2 = ax6_twin.plot(df_sorted['epoch'], dormant_rolling, 'b-', linewidth=2,
                          label=f'Dormant Prop (rolling avg, window={window})')
    
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Memorization Score', color='red')
    ax6_twin.set_ylabel('Dormant Unit Proportion', color='blue')
    ax6.set_title('Rolling Averages Over Training', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    return fig

def create_detailed_analysis_plot(df):
    """Create detailed analysis with binned statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Binned analysis: Memo score by dormant proportion bins
    ax1 = axes[0, 0]
    df['dormant_bin'] = pd.cut(df['top_dormant_prop'], bins=10)
    binned_stats = df.groupby('dormant_bin')['avg_memo_score'].agg(['mean', 'std', 'count'])
    bin_centers = [interval.mid for interval in binned_stats.index]
    
    ax1.errorbar(bin_centers, binned_stats['mean'], yerr=binned_stats['std'], 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Dormant Unit Proportion (Bin Centers)')
    ax1.set_ylabel('Average Memorization Score')
    ax1.set_title('Memorization Score vs Dormant Proportion\n(Binned Analysis)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time evolution with phases
    ax2 = axes[0, 1]
    # Define training phases
    early = df[df['epoch'] <= 1000]
    mid = df[(df['epoch'] > 1000) & (df['epoch'] <= 3000)]
    late = df[df['epoch'] > 3000]
    
    ax2.scatter(early['top_dormant_prop'], early['avg_memo_score'], 
               alpha=0.6, s=15, label=f'Early (1-1000): {len(early)} epochs', color='green')
    ax2.scatter(mid['top_dormant_prop'], mid['avg_memo_score'], 
               alpha=0.6, s=15, label=f'Mid (1001-3000): {len(mid)} epochs', color='orange')
    ax2.scatter(late['top_dormant_prop'], late['avg_memo_score'], 
               alpha=0.6, s=15, label=f'Late (3001-4000): {len(late)} epochs', color='red')
    
    ax2.set_xlabel('Dormant Unit Proportion')
    ax2.set_ylabel('Memorization Score')
    ax2.set_title('Training Phase Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Extreme values analysis
    ax3 = axes[1, 0]
    high_memo = df[df['avg_memo_score'] > 0.9]
    low_memo = df[df['avg_memo_score'] < 0.1]
    mid_memo = df[(df['avg_memo_score'] >= 0.4) & (df['avg_memo_score'] <= 0.6)]
    
    data_to_plot = [high_memo['top_dormant_prop'], mid_memo['top_dormant_prop'], low_memo['top_dormant_prop']]
    labels = [f'High Memo (>0.9)\n{len(high_memo)} epochs', 
              f'Mid Memo (0.4-0.6)\n{len(mid_memo)} epochs',
              f'Low Memo (<0.1)\n{len(low_memo)} epochs']
    
    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['red', 'yellow', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_ylabel('Dormant Unit Proportion')
    ax3.set_title('Dormant Proportions by Memorization Level', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample frequency analysis
    ax4 = axes[1, 1]
    sample_counts = df['top_sample_id'].value_counts().head(10)
    
    bars = ax4.bar(range(len(sample_counts)), sample_counts.values)
    ax4.set_xlabel('Sample ID (Top 10 Most Frequent)')
    ax4.set_ylabel('Number of Epochs as Top Dormant')
    ax4.set_title('Most Frequently Top-Dormant Samples', fontweight='bold')
    ax4.set_xticks(range(len(sample_counts)))
    ax4.set_xticklabels([f'{int(sid)}' for sid in sample_counts.index], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(sample_counts.values):
        ax4.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def print_summary_statistics(df):
    """Print detailed summary statistics."""
    print("\n" + "="*70)
    print("DETAILED STATISTICAL SUMMARY")
    print("="*70)
    
    print(f"\nOverall Statistics (4000 epochs):")
    print(f"  Average memorization score: {df['avg_memo_score'].mean():.4f} ± {df['avg_memo_score'].std():.4f}")
    print(f"  Average dormant proportion: {df['top_dormant_prop'].mean():.4f} ± {df['top_dormant_prop'].std():.4f}")
    print(f"  Correlation coefficient: {df['top_dormant_prop'].corr(df['avg_memo_score']):.4f}")
    
    # Quartile analysis
    memo_q25, memo_q50, memo_q75 = df['avg_memo_score'].quantile([0.25, 0.5, 0.75])
    dormant_q25, dormant_q50, dormant_q75 = df['top_dormant_prop'].quantile([0.25, 0.5, 0.75])
    
    print(f"\nMemorization Score Quartiles:")
    print(f"  Q1 (25%): {memo_q25:.4f}")
    print(f"  Q2 (50%): {memo_q50:.4f}")  
    print(f"  Q3 (75%): {memo_q75:.4f}")
    
    print(f"\nDormant Proportion Quartiles:")
    print(f"  Q1 (25%): {dormant_q25:.4f}")
    print(f"  Q2 (50%): {dormant_q50:.4f}")
    print(f"  Q3 (75%): {dormant_q75:.4f}")
    
    # Most common samples
    top_samples = df['top_sample_id'].value_counts().head(5)
    print(f"\nMost frequently top-dormant samples:")
    for sample_id, count in top_samples.items():
        sample_data = df[df['top_sample_id'] == sample_id]
        avg_memo = sample_data['avg_memo_score'].mean()
        avg_dormant = sample_data['top_dormant_prop'].mean()
        print(f"  Sample {int(sample_id)}: {count} epochs, avg memo {avg_memo:.3f}, avg dormant {avg_dormant:.3f}")

def main():
    """Main visualization function."""
    
    # Load data
    df = load_analysis_data()
    if df is None:
        return
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create comprehensive visualization
    print("\nCreating comprehensive visualization...")
    fig1 = create_comprehensive_visualization(df)
    
    # Save comprehensive plot
    output_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system"
    plot1_path = f"{output_dir}/dormant_memo_comprehensive_analysis.png"
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {plot1_path}")
    
    # Create detailed analysis
    print("Creating detailed analysis plots...")
    fig2 = create_detailed_analysis_plot(df)
    
    # Save detailed plot
    plot2_path = f"{output_dir}/dormant_memo_detailed_analysis.png"
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {plot2_path}")
    
    print("Plots have been saved successfully!")
    print("Since we're using non-interactive backend, plots are saved as files instead of displayed.")

if __name__ == "__main__":
    main()