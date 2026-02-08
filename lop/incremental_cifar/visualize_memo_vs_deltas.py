#!/usr/bin/env python3
"""
Visualization script for memo scores vs delta metrics from windowed plasticity data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def load_data(csv_path):
    """Load the windowed plasticity data."""
    return pd.read_csv(csv_path)

def create_memo_delta_plots(df, output_dir=None):
    """Create scatter plots of memo scores vs each delta metric."""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    delta_columns = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    titles = ['Memo Score vs Delta Loss', 'Memo Score vs Delta Grad Norm', 'Memo Score vs Delta Dormant']
    
    for i, (delta_col, title) in enumerate(zip(delta_columns, titles)):
        ax = axes[i]
        
        # Create scatter plot
        scatter = ax.scatter(df['memo_score'], df[delta_col], 
                           alpha=0.6, s=1, c=df['task_number'], 
                           cmap='viridis')
        
        ax.set_xlabel('Memo Score')
        ax.set_ylabel(delta_col.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for task number
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Task Number')
        
        # Add correlation coefficient
        corr = df['memo_score'].corr(df[delta_col])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'memo_vs_deltas.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()
    return fig

def create_detailed_analysis(df, output_dir=None):
    """Create additional detailed analysis plots."""
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Memo score distribution
    axes[0, 0].hist(df['memo_score'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Memo Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Memo Scores')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Delta distributions
    delta_columns = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    colors = ['blue', 'red', 'green']
    
    for i, (delta_col, color) in enumerate(zip(delta_columns, colors)):
        if i < 3:  # Only plot first 3 in remaining subplots
            row = (i + 1) // 2
            col = (i + 1) % 2
            axes[row, col].hist(df[delta_col], bins=50, alpha=0.7, 
                               color=color, edgecolor='black')
            axes[row, col].set_xlabel(delta_col.replace('_', ' ').title())
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'Distribution of {delta_col.replace("_", " ").title()}')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {output_path}")
    
    plt.show()
    
    # Correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_columns = ['memo_score'] + delta_columns + ['loss_pre', 'loss_post', 'dormant_prop_pre', 'dormant_prop_post']
    corr_matrix = df[corr_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, fmt='.3f')
    ax.set_title('Correlation Matrix: Memo Score vs Delta Metrics')
    
    if output_dir:
        output_path = Path(output_dir) / 'correlation_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {output_path}")
    
    plt.show()
    
    return fig

def print_summary_stats(df):
    """Print summary statistics."""
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"Total samples: {len(df):,}")
    print(f"Unique tasks: {df['task_number'].nunique()}")
    print(f"Unique classes: {df['class_label'].nunique()}")
    
    print("\nMemo Score Statistics:")
    print(df['memo_score'].describe())
    
    print("\nDelta Metrics Statistics:")
    delta_columns = ['delta_loss', 'delta_grad_norm', 'delta_dormant']
    for col in delta_columns:
        print(f"\n{col}:")
        print(df[col].describe())
    
    print("\nCorrelations with Memo Score:")
    for col in delta_columns:
        corr = df['memo_score'].corr(df[col])
        print(f"{col}: {corr:.4f}")

def main():
    # Data path
    csv_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/windowed_plasticity_data.csv"
    
    # Output directory for plots
    output_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading data...")
    df = load_data(csv_path)
    
    print("Creating memo vs delta plots...")
    create_memo_delta_plots(df, output_dir)
    
    print("Creating detailed analysis...")
    create_detailed_analysis(df, output_dir)
    
    print_summary_stats(df)

if __name__ == "__main__":
    main()