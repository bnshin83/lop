#!/usr/bin/env python3
"""
Quick test of the fixed sample order extraction by running a minimal experiment
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incremental_cifar_memo_ordered_experiment import IncrementalCIFARMemoOrderedExperiment

def test_fixed_experiment():
    """Test the fixed experiment with a minimal run"""
    
    print("=" * 80)
    print("TESTING FIXED SAMPLE ORDER EXTRACTION")
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
        "num_workers": 0  # Use 0 to avoid multiprocessing issues during testing
    }
    
    # Create temporary results directory
    results_dir = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/test_fixed_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Running minimal test with fixed sample order extraction...")
    print(f"Results dir: {results_dir}")
    
    try:
        # Create experiment instance with same parameters as the random sampling experiment
        experiment = IncrementalCIFARMemoOrderedExperiment(
            exp_params=exp_params,
            results_dir=results_dir,
            run_index=0,  # Same run_index as the original experiment
            verbose=True,
            memo_order="low_to_high",
            gpu_id=0,
            no_ordering=True,  # This enables random shuffling like the original experiment
            scratch=True,
            epochs_per_task=1,  # Just 1 epoch for testing
            incremental_epochs=False,
            start_task=19,  # Single task with all 100 classes
            max_tasks=1,
            class_order="random"
        )
        
        print(f"Experiment created successfully")
        print(f"Random seed: {experiment.random_seed}")
        print(f"No ordering (shuffle): {experiment.no_ordering}")
        
        # Test just the data loading part (which will extract sample order)
        print(f"\nTesting data loading and sample order extraction...")
        training_data, training_dataloader = experiment.get_data(train=True, validation=False)
        val_data, val_dataloader = experiment.get_data(train=True, validation=True)
        
        print(f"Data loaded successfully!")
        print(f"Training data size: {len(training_data)}")
        print(f"Validation data size: {len(val_data)}")
        
        # Check if sample order files were created
        sample_order_files = [f for f in os.listdir(results_dir) if f.startswith("actual_sample_order")]
        if sample_order_files:
            print(f"\n✓ Sample order file created: {sample_order_files}")
            
            # Read and display the first few samples
            sample_file = os.path.join(results_dir, sample_order_files[0])
            import pandas as pd
            df = pd.read_csv(sample_file)
            print(f"Sample order file has {len(df)} samples")
            print(f"First 10 sample IDs: {df['sample_id'].head(10).tolist()}")
            
            return df['sample_id'].tolist()
        else:
            print(f"✗ No sample order file was created")
            return None
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_fixed_experiment()