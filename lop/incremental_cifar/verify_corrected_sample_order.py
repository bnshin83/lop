#!/usr/bin/env python3
"""
Verify the corrected sample order generation using the proper experiment seed
"""

import torch
import pandas as pd
from mlproj_manager.util import get_random_seeds

def test_sample_order_extraction():
    """Test sample order extraction with correct vs incorrect seed"""
    
    # Get actual experiment seed
    random_seeds = get_random_seeds()
    experiment_seed = int(random_seeds[0])  # run_index = 0, convert to int
    
    print("=" * 80)
    print("VERIFYING SAMPLE ORDER EXTRACTION")
    print("=" * 80)
    print(f"Experiment seed (should be used): {experiment_seed}")
    
    # Simulate train indices (class-grouped as in experiment)
    train_indices = []
    for class_id in range(100):
        class_start = class_id * 500  # Each class has samples 500*class_id to 500*(class_id+1)-1
        # Take samples 50-499 for training (450 samples per class, skipping first 50 for validation)
        class_train_indices = list(range(class_start + 50, class_start + 500))
        train_indices.extend(class_train_indices)
    
    train_indices = torch.tensor(train_indices, dtype=torch.int32)
    print(f"Train indices length: {len(train_indices)}")
    print(f"Train indices first 10: {train_indices[:10].tolist()}")
    
    # Test with WRONG seed (42) - this is the current bug
    print(f"\n1. Using WRONG seed (42) - current buggy behavior:")
    generator_wrong = torch.Generator()
    generator_wrong.manual_seed(42)
    shuffle_order_wrong = torch.randperm(len(train_indices), generator=generator_wrong)
    wrong_order = [train_indices[i].item() for i in shuffle_order_wrong[:10]]
    print(f"   Wrong order first 10: {wrong_order}")
    
    # Test with CORRECT seed (37542) - this should match DataLoader
    print(f"\n2. Using CORRECT seed ({experiment_seed}) - should match DataLoader:")
    generator_correct = torch.Generator()
    generator_correct.manual_seed(experiment_seed)
    shuffle_order_correct = torch.randperm(len(train_indices), generator=generator_correct)
    correct_order = [train_indices[i].item() for i in shuffle_order_correct[:10]]
    print(f"   Correct order first 10: {correct_order}")
    
    # Load the extracted order from actual experiment
    print(f"\n3. Extracted order from actual experiment:")
    try:
        extracted_df = pd.read_csv('/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_random7_single_200/actual_sample_order_task_19_epoch_0.csv')
        extracted_order = extracted_df['sample_id'].tolist()[:10]
        print(f"   Extracted order first 10: {extracted_order}")
        
        # Check which one matches
        if wrong_order == extracted_order:
            print(f"\n✓ WRONG seed (42) matches extracted order - confirms the bug!")
        elif correct_order == extracted_order:
            print(f"\n✓ CORRECT seed ({experiment_seed}) matches extracted order - bug is fixed!")
        else:
            print(f"\n✗ Neither matches - need more investigation")
            
    except Exception as e:
        print(f"   Error loading extracted order: {e}")
    
    return experiment_seed

if __name__ == "__main__":
    test_sample_order_extraction()