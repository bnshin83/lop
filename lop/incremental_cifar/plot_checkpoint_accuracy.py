#!/usr/bin/env python3
"""
Script to plot test accuracy over tasks (checkpoints) from CSV file
Usage: python plot_checkpoint_accuracy.py <csv_file_path>
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def plot_accuracy_over_tasks(csv_file_path, save_plot=True):
    """
    Plot test accuracy over tasks from checkpoint evaluation CSV file
    
    Args:
        csv_file_path (str): Path to the CSV file containing checkpoint evaluations
        save_plot (bool): Whether to save the plot as PNG file
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded data from: {csv_file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: File not found - {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Sort by epoch to ensure proper ordering
    df = df.sort_values('epoch')
    
    # Calculate number of tasks (assuming 5 classes per task based on the data)
    # num_classes_evaluated gives us cumulative classes, so task = num_classes_evaluated / 5
    df['task'] = df['num_classes_evaluated'] / 5
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['task'], df['test_accuracy'], 'b-o', linewidth=2, markersize=4, alpha=0.7)
    
    # Add accuracy values as labels on each point
    for i, (task, accuracy) in enumerate(zip(df['task'], df['test_accuracy'])):
        plt.annotate(f'{accuracy:.3f}', (task, accuracy), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=8, alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Task Number', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy vs Task Number\n(Incremental CIFAR-100)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visualization
    plt.ylim(0, 1.0)
    
    # Add some statistics as text
    max_acc = df['test_accuracy'].max()
    min_acc = df['test_accuracy'].min()
    final_acc = df['test_accuracy'].iloc[-1]
    
    stats_text = f'Max Accuracy: {max_acc:.3f}\nFinal Accuracy: {final_acc:.3f}\nMin Accuracy: {min_acc:.3f}'
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        output_dir = os.path.dirname(csv_file_path)
        output_file = os.path.join(output_dir, 'test_accuracy_over_tasks.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Number of checkpoints: {len(df)}")
    print(f"Task range: {df['task'].min():.1f} to {df['task'].max():.1f}")
    print(f"Accuracy range: {min_acc:.3f} to {max_acc:.3f}")
    print(f"Final accuracy: {final_acc:.3f}")
    print(f"Accuracy drop from max: {max_acc - final_acc:.3f}")

def main():
    # Ask user for CSV file path
    csv_file_path = input("Enter the path to the CSV file: ").strip()
    
    # Remove quotes if user included them
    if csv_file_path.startswith('"') and csv_file_path.endswith('"'):
        csv_file_path = csv_file_path[1:-1]
    if csv_file_path.startswith("'") and csv_file_path.endswith("'"):
        csv_file_path = csv_file_path[1:-1]
    
    if not csv_file_path:
        print("Error: No file path provided")
        return
    
    if not os.path.exists(csv_file_path):
        print(f"Error: File does not exist - {csv_file_path}")
        return
    
    plot_accuracy_over_tasks(csv_file_path, save_plot=True)

if __name__ == '__main__':
    main()