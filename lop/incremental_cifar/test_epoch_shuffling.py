#!/usr/bin/env python3
"""
Test whether PyTorch DataLoader actually shuffles differently across epochs
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mlproj_manager.util import get_random_seeds

def test_dataloader_epoch_shuffling():
    """Test if DataLoader shuffles differently across epochs"""
    
    print("=" * 80)
    print("TESTING DATALOADER EPOCH-TO-EPOCH SHUFFLING BEHAVIOR")
    print("=" * 80)
    
    # Set up like the experiment
    random_seeds = get_random_seeds()
    experiment_seed = int(random_seeds[0])
    print(f"Experiment seed: {experiment_seed}")
    
    # Set the seed exactly like the experiment
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    
    # Create a simple dataset
    data = torch.arange(20)  # Simple data: [0, 1, 2, ..., 19]
    labels = torch.zeros(20)  # Dummy labels
    dataset = TensorDataset(data, labels)
    
    print(f"Original data order: {data.tolist()}")
    
    # Create DataLoader with shuffle=True (like random experiment)
    print(f"\n1. Testing DataLoader with shuffle=True:")
    dataloader_shuffled = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    
    # Test multiple epochs
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        epoch_order = []
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader_shuffled):
            epoch_order.extend(batch_data.tolist())
        print(f"  Full epoch order: {epoch_order}")
    
    # Create DataLoader with shuffle=False (like predetermined experiment)
    print(f"\n2. Testing DataLoader with shuffle=False:")
    dataloader_fixed = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
    
    # Test multiple epochs
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        epoch_order = []
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader_fixed):
            epoch_order.extend(batch_data.tolist())
        print(f"  Full epoch order: {epoch_order}")
    
    # Test with explicit generator
    print(f"\n3. Testing with explicit generator:")
    generator = torch.Generator()
    generator.manual_seed(experiment_seed)
    
    dataloader_gen = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0, generator=generator)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1} (with generator):")
        epoch_order = []
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader_gen):
            epoch_order.extend(batch_data.tolist())
        print(f"  Full epoch order: {epoch_order}")
    
    # Test what happens if we recreate the DataLoader each time
    print(f"\n4. Testing by recreating DataLoader each epoch:")
    for epoch in range(3):
        # Reset seed before creating new DataLoader
        torch.manual_seed(experiment_seed)
        new_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
        
        print(f"\nEpoch {epoch + 1} (new DataLoader):")
        epoch_order = []
        for batch_idx, (batch_data, batch_labels) in enumerate(new_dataloader):
            epoch_order.extend(batch_data.tolist())
        print(f"  Full epoch order: {epoch_order}")

if __name__ == "__main__":
    test_dataloader_epoch_shuffling()