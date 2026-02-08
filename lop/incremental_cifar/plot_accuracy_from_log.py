#!/usr/bin/env python3
"""
Plot accuracy evolution from incremental CIFAR log files.
Extracts online accuracy (last per epoch), test accuracy, and validation accuracy.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def parse_log_file(log_path):
    """
    Parse log file to extract accuracy values per epoch.
    
    Returns:
        epochs, online_accuracies, test_accuracies, validation_accuracies
    """
    epochs = []
    online_accuracies = []
    test_accuracies = []
    validation_accuracies = []
    
    current_epoch = None
    current_online_accs = []  # Store all online accuracies for current epoch
    current_test_acc = None
    current_val_acc = None
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Match epoch number (with optional base/high_memo suffix)
            # Support both old format "Epoch number:" and new format "[GPU X] [Task Y] Epoch Z"
            epoch_match = re.search(r'(?:Epoch number:\s+(\d+)(?:\s+\([^)]+\))?|\[GPU\s+\d+\]\s+\[Task\s+\d+\]\s+Epoch\s+(\d+))', line)
            if epoch_match:
                # Save previous epoch data if we have test and validation (online is optional)
                if (current_epoch is not None and 
                    current_test_acc is not None and 
                    current_val_acc is not None):
                    
                    epochs.append(current_epoch)
                    # Use online accuracy if available, otherwise use NaN for plotting
                    if len(current_online_accs) > 0:
                        online_accuracies.append(min(1.0, max(current_online_accs)))
                    else:
                        online_accuracies.append(float('nan'))  # NaN for missing data
                    test_accuracies.append(current_test_acc)
                    validation_accuracies.append(current_val_acc)
                
                # Start new epoch - extract from either group (1 or 2)
                current_epoch = int(epoch_match.group(1) or epoch_match.group(2))
                current_online_accs = []
                current_test_acc = None
                current_val_acc = None
                continue
            
            # Match online accuracy (both regular and end-of-epoch)
            online_match = re.search(r'(?:Online accuracy|End-of-epoch online accuracy)(?:\s+\([^)]+\))?:\s+([\d.]+)', line)
            if online_match:
                online_val = float(online_match.group(1))
                current_online_accs.append(online_val)
                # Debug: Print values > 1.0
                if online_val > 1.0:
                    print(f"Debug - Found online accuracy > 1.0: {online_val} at epoch {current_epoch}")
                continue
            
            # Match test accuracy (with optional suffix in parentheses)
            test_match = re.search(r'test accuracy(?:\s+\([^)]+\))?:\s+([\d.]+)', line)
            if test_match:
                current_test_acc = float(test_match.group(1))
                continue
            
            # Match validation accuracy (with optional suffix in parentheses)
            val_match = re.search(r'validation accuracy(?:\s+\([^)]+\))?:\s+([\d.]+)', line)
            if val_match:
                current_val_acc = float(val_match.group(1))
                continue
    
    # Don't forget the last epoch
    if (current_epoch is not None and 
        current_test_acc is not None and 
        current_val_acc is not None):
        
        epochs.append(current_epoch)
        # Use online accuracy if available, otherwise use NaN for plotting
        if len(current_online_accs) > 0:
            online_accuracies.append(min(1.0, max(current_online_accs)))
        else:
            online_accuracies.append(float('nan'))  # NaN for missing data
        test_accuracies.append(current_test_acc)
        validation_accuracies.append(current_val_acc)
    
    return np.array(epochs), np.array(online_accuracies), np.array(test_accuracies), np.array(validation_accuracies)

def plot_accuracies(epochs, online_accs, test_accs, val_accs, log_path):
    """Plot the three accuracy curves."""
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the three accuracy curves (handle NaN values gracefully)
    # Only plot online accuracy where we have valid data
    valid_online_mask = ~np.isnan(online_accs)
    if np.any(valid_online_mask):
        plt.plot(epochs[valid_online_mask], online_accs[valid_online_mask], 'b-', 
                label='Online Accuracy (Max per Epoch)', linewidth=2, alpha=0.8)
        
        # Mark points where online accuracy was over 1.0
        over_one_mask = (online_accs > 1.0) & valid_online_mask
        if np.any(over_one_mask):
            plt.scatter(epochs[over_one_mask], online_accs[over_one_mask], 
                       color='red', marker='x', s=50, label='Online Acc > 1.0 (Bug)', zorder=5)
    
    # Always plot test and validation (should not have NaN values)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy', linewidth=2, alpha=0.8)
    
    # Add vertical lines at class addition points (every 200 epochs)
    max_epoch = epochs[-1] if len(epochs) > 0 else 4000
    for class_add_epoch in range(200, max_epoch, 200):
        if class_add_epoch <= max_epoch:
            plt.axvline(x=class_add_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add horizontal line at y=1.0 for reference (before ylim setting)
    plt.axhline(y=1.0, color='black', linestyle=':', alpha=0.7, linewidth=1)
    
    # Set y-axis limits (exclude NaN values)
    valid_values = []
    if np.any(valid_online_mask):
        valid_values.extend(online_accs[valid_online_mask])
    valid_values.extend(test_accs)
    valid_values.extend(val_accs)
    
    if len(valid_values) > 0:
        all_values = np.array(valid_values)
        y_min = max(0, np.min(all_values) - 0.05)  # Start at 0 or slightly below min
        y_max = 1.2  # Cap at 1.2
    else:
        y_min, y_max = 0, 1.2
    
    # Debug: Print ylim values
    print(f"Debug - Setting ylim to: ({y_min:.3f}, {y_max:.3f})")
    print(f"Debug - Max value in data: {np.max(all_values):.3f}")
    
    plt.ylim(y_min, y_max)
    
    # Force ylim again after other plot elements
    current_ylim = plt.gca().get_ylim()
    print(f"Debug - Current ylim after setting: {current_ylim}")
    if current_ylim[1] < y_max:
        print(f"Debug - Forcing ylim again because current max {current_ylim[1]:.3f} < desired max {y_max:.3f}")
        plt.gca().set_ylim(y_min, y_max)
    
    # Customize plot
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Accuracy Evolution Over Training\nLog: {os.path.basename(log_path)}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation about problematic values
    if np.any(over_one_mask):
        num_over_one = np.sum(over_one_mask)
        max_online = np.max(online_accs)
        plt.text(0.02, 0.98, f'Note: {num_over_one} online accuracy values > 1.0 (max: {max_online:.2f})', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Final ylim check after all plot elements
    final_ylim = plt.gca().get_ylim()
    print(f"Debug - Final ylim check: {final_ylim}")
    if final_ylim[1] < y_max:
        print(f"Debug - FINAL FORCE: Setting ylim to ({y_min:.3f}, {y_max:.3f})")
        plt.gca().set_ylim(y_min, y_max)
        print(f"Debug - After final force: {plt.gca().get_ylim()}")
    
    plt.tight_layout()
    
    # Save plot
    output_path = log_path.replace('.out', '_accuracy_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total epochs processed: {len(epochs)}")
    
    # Online accuracy statistics (handle NaN values)
    if np.any(valid_online_mask):
        valid_online_accs = online_accs[valid_online_mask]
        print(f"Online accuracy - Min: {np.min(valid_online_accs):.3f}, Max: {np.max(valid_online_accs):.3f}, Mean: {np.mean(valid_online_accs):.3f}")
        print(f"Online accuracy data available for: {np.sum(valid_online_mask)} out of {len(epochs)} epochs ({np.sum(valid_online_mask)/len(epochs)*100:.1f}%)")
        
        over_one_mask = (online_accs > 1.0) & valid_online_mask
        if np.any(over_one_mask):
            print(f"Online accuracy values > 1.0: {np.sum(over_one_mask)} out of {np.sum(valid_online_mask)} ({np.sum(over_one_mask)/np.sum(valid_online_mask)*100:.1f}%)")
    else:
        print("Online accuracy - No data available")
    
    print(f"Test accuracy - Min: {np.min(test_accs):.3f}, Max: {np.max(test_accs):.3f}, Mean: {np.mean(test_accs):.3f}")
    print(f"Validation accuracy - Min: {np.min(val_accs):.3f}, Max: {np.max(val_accs):.3f}, Mean: {np.mean(val_accs):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy evolution from incremental CIFAR log files")
    parser.add_argument("log_file", type=str, help="Path to log file")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    print(f"Parsing log file: {args.log_file}")
    epochs, online_accs, test_accs, val_accs = parse_log_file(args.log_file)
    
    if len(epochs) == 0:
        print("Error: No epoch data found in log file")
        return
    
    print(f"Found data for {len(epochs)} epochs")
    
    # Debug: Print some sample values and statistics
    print(f"Debug - First 10 online accuracies: {online_accs[:10]}")
    print(f"Debug - Online accuracy range: {np.min(online_accs):.3f} to {np.max(online_accs):.3f}")
    print(f"Debug - Values > 1.0: {np.sum(online_accs > 1.0)} out of {len(online_accs)}")
    if np.any(online_accs > 1.0):
        print(f"Debug - Sample values > 1.0: {online_accs[online_accs > 1.0][:10]}")
    
    plot_accuracies(epochs, online_accs, test_accs, val_accs, args.log_file)

if __name__ == "__main__":
    main()