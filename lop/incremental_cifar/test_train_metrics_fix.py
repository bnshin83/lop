#!/usr/bin/env python3
"""
Test the train metrics fix
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incremental_cifar_memo_ordered_experiment import IncrementalCIFARMemoOrderedExperiment

def test_train_metrics_fix():
    """Test that the train metrics fix works"""
    
    print("=" * 80)
    print("TESTING TRAIN METRICS FIX")
    print("=" * 80)
    
    # Create a minimal config for testing
    exp_params = {
        "data_path": "/tmp",
        "stepsize": 0.1,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "noise_std": 0.0,
        "use_cbp": False,
        "reset_head": False,
        "reset_network": False,
        "early_stopping": True,
        "num_workers": 0  # Use 0 to avoid multiprocessing issues
    }
    
    # Create temporary results directory
    results_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/test_train_metrics_fix"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Testing with minimal experiment (2 epochs)...")
    
    try:
        # Create experiment instance
        experiment = IncrementalCIFARMemoOrderedExperiment(
            exp_params=exp_params,
            results_dir=results_dir,
            run_index=0,  
            verbose=True,
            memo_order="low_to_high",
            gpu_id=0,
            no_ordering=True,  # Random shuffling
            scratch=True,
            epochs_per_task=2,  # Just 2 epochs for testing
            incremental_epochs=False,
            start_task=19,  # Single task with all 100 classes
            max_tasks=1,
            class_order="random"
        )
        
        print(f"Experiment created successfully")
        print(f"Testing train metrics with 2 epochs...")
        
        # Run the experiment
        experiment.run()
        
        # Check if train metrics were saved correctly
        task_dir = os.path.join(results_dir, "no_ordering_scratch_fixed_2_experiment", "task_19")
        train_acc_file = os.path.join(task_dir, "train_accuracy_per_epoch_task_19.csv")
        train_loss_file = os.path.join(task_dir, "train_loss_per_epoch_task_19.csv")
        
        success = True
        
        if os.path.exists(train_acc_file):
            import pandas as pd
            train_acc_df = pd.read_csv(train_acc_file)
            print(f"✓ Train accuracy CSV exists with {len(train_acc_df)} rows")
            print(f"  Values: {train_acc_df['Value'].tolist()}")
            
            # Check if values are non-zero
            non_zero_count = (train_acc_df['Value'] != 0.0).sum()
            if non_zero_count > 0:
                print(f"✓ SUCCESS: {non_zero_count}/{len(train_acc_df)} values are non-zero!")
            else:
                print(f"✗ FAIL: All values are still zero")
                success = False
        else:
            print(f"✗ Train accuracy CSV file not found: {train_acc_file}")
            success = False
            
        if os.path.exists(train_loss_file):
            train_loss_df = pd.read_csv(train_loss_file)
            print(f"✓ Train loss CSV exists with {len(train_loss_df)} rows")
            print(f"  Values: {train_loss_df['Value'].tolist()}")
            
            # Check if values are non-zero
            non_zero_count = (train_loss_df['Value'] != 0.0).sum()
            if non_zero_count > 0:
                print(f"✓ SUCCESS: {non_zero_count}/{len(train_loss_df)} values are non-zero!")
            else:
                print(f"✗ FAIL: All values are still zero")
                success = False
        else:
            print(f"✗ Train loss CSV file not found: {train_loss_file}")
            success = False
        
        if success:
            print(f"\n🎉 TRAIN METRICS FIX SUCCESSFUL!")
            print(f"Train accuracy and loss are now being saved correctly.")
        else:
            print(f"\n❌ TRAIN METRICS FIX FAILED!")
            print(f"Train metrics are still showing as zeros.")
            
        return success
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_train_metrics_fix()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")