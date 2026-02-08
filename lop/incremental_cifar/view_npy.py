import numpy as np
import os

# Load the .npy file
file_path = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/test_accuracy_per_epoch/index-0.npy'
data = np.load(file_path, allow_pickle=True)

# Print basic information
print(f"File: {file_path}")
print(f"Type: {type(data)}")
print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
print(f"Data type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")

# Print first few elements (or summary if it's a large array)
print("\nFirst few elements:")
if hasattr(data, 'shape') and len(data.shape) > 0 and data.size > 10:
    print(data.flat[:100])  # First 10 elements for large arrays
else:
    print(data)
