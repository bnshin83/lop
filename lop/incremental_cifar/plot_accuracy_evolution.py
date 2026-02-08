#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_accuracy_evolution():
    """Plot test and validation accuracy over epochs"""
    
    # Load the accuracy data
    test_acc_path = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system/test_accuracy_per_epoch/index-0.npy'
    val_acc_path = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results/base_deep_learning_system/validation_accuracy_per_epoch/index-0.npy'
    
    print(f'Loading test accuracy from: {test_acc_path}')
    print(f'Loading validation accuracy from: {val_acc_path}')
    
    # Load data
    test_accuracy = np.load(test_acc_path)
    val_accuracy = np.load(val_acc_path)
    
    # Create epoch array
    epochs = np.arange(1, len(test_accuracy) + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot both accuracies
    plt.plot(epochs, test_accuracy, label='Test Accuracy', color='blue', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='red', linewidth=2, alpha=0.8)
    
    # Add vertical lines at class addition points (every 200 epochs for 20 classes)
    class_addition_epochs = np.arange(200, len(epochs), 200)
    for epoch in class_addition_epochs:
        if epoch < len(epochs):
            plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Test and Validation Accuracy Evolution Over Epochs\n(Incremental CIFAR-100)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add text annotation about class additions
    plt.text(0.02, 0.98, 'Vertical lines indicate new class additions (every 200 epochs)', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show some statistics
    final_test_acc = test_accuracy[-1]
    final_val_acc = val_accuracy[-1]
    max_test_acc = np.max(test_accuracy)
    max_val_acc = np.max(val_accuracy)
    
    stats_text = f'Final Test Acc: {final_test_acc:.3f}\nFinal Val Acc: {final_val_acc:.3f}\nMax Test Acc: {max_test_acc:.3f}\nMax Val Acc: {max_val_acc:.3f}'
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, 'accuracy_evolution_over_epochs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {output_path}')
    
    # Also save as SVG for better quality
    output_path_svg = os.path.join(plots_dir, 'accuracy_evolution_over_epochs.svg')
    plt.savefig(output_path_svg, bbox_inches='tight')
    print(f'Plot also saved as SVG: {output_path_svg}')
    
    plt.show()
    
    return test_accuracy, val_accuracy, epochs

if __name__ == "__main__":
    test_acc, val_acc, epochs = plot_accuracy_evolution()