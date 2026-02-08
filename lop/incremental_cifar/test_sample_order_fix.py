#!/usr/bin/env python3
"""
Test the sample order extraction fix
"""

import torch
import pandas as pd
from mlproj_manager.util import get_random_seeds

def test_corrected_extraction():
    """Test the corrected sample order extraction logic"""
    
    print("=" * 80)
    print("TESTING SAMPLE ORDER EXTRACTION FIX")
    print("=" * 80)
    
    # Get actual experiment seed (run_index = 0)
    random_seeds = get_random_seeds()
    experiment_seed = int(random_seeds[0])
    print(f"Experiment seed: {experiment_seed}")
    
    # Simulate the train indices (class-grouped like in experiment)
    # This matches what get_validation_and_train_indices creates
    train_indices = []
    for class_id in range(100):
        class_start = class_id * 500
        # Skip first 50 samples of each class (reserved for validation)
        class_train_indices = list(range(class_start + 50, class_start + 500))
        train_indices.extend(class_train_indices)
    
    train_indices = torch.tensor(train_indices, dtype=torch.int32)
    print(f"Train indices length: {len(train_indices)}")
    print(f"Train indices first 10: {train_indices[:10].tolist()}")
    
    # Simulate the FIXED extraction logic using actual experiment seed
    print(f"\nUsing CORRECTED seed ({experiment_seed}):")
    generator = torch.Generator()
    generator.manual_seed(experiment_seed)  # This is the fix!
    
    # Generate the same permutation that RandomSampler would create
    n_samples = len(train_indices)
    shuffle_indices = torch.randperm(n_samples, generator=generator).tolist()
    
    # Map to actual sample IDs
    corrected_order = [train_indices[i].item() for i in shuffle_indices]
    print(f"Corrected order first 10: {corrected_order[:10]}")
    
    # Load the extracted order from the actual experiment (uses old buggy seed 42)
    print(f"\nComparing with extracted order from experiment:")
    try:
        extracted_df = pd.read_csv('/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_random7_single_200/actual_sample_order_task_19_epoch_0.csv')
        extracted_order = extracted_df['sample_id'].tolist()
        print(f"Extracted order first 10:  {extracted_order[:10]}")
        
        if corrected_order[:10] == extracted_order[:10]:
            print(f"\n✓ SUCCESS: Corrected order matches extracted order!")
        else:
            print(f"\n✗ Orders still don't match (expected - old experiment used buggy seed)")
            print("The fix is correct, but we need to run a new experiment to verify")
        
        # Show the difference explicitly
        print(f"\nDetailed comparison:")
        for i in range(10):
            status = "✓" if corrected_order[i] == extracted_order[i] else "✗"
            print(f"  {status} Position {i}: Corrected={corrected_order[i]}, Old={extracted_order[i]}")
            
    except Exception as e:
        print(f"Error loading extracted order: {e}")
    
    # Now test with the OLD buggy seed (42) to confirm it would match the extracted order
    print(f"\n" + "=" * 80)
    print("TESTING OLD BUGGY BEHAVIOR (seed=42)")
    print("=" * 80)
    
    generator_old = torch.Generator()
    generator_old.manual_seed(42)  # Old buggy seed
    shuffle_indices_old = torch.randperm(n_samples, generator=generator_old).tolist()
    old_buggy_order = [train_indices[i].item() for i in shuffle_indices_old]
    print(f"Old buggy order first 10: {old_buggy_order[:10]}")
    
    if 'extracted_order' in locals():
        if old_buggy_order[:10] == extracted_order[:10]:
            print(f"✓ OLD buggy order DOES match extracted order - confirms the bug!")
        else:
            print(f"✗ Even old buggy order doesn't match - more investigation needed")
    
    return corrected_order

if __name__ == "__main__":
    test_corrected_extraction()