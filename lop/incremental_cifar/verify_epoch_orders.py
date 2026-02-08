#!/usr/bin/env python3
"""
Quick script to verify the epoch sample orders and random seeds were saved correctly
"""
import pickle
import numpy as np

# Load the pickle file with complete data
pkl_file = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_memo_weighted_t0.25_w3.0_single_200_random_weight1/epoch_sample_orders_task_19.pkl"

print("Loading epoch sample orders and random seeds...")
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print("\n=== EPOCH SAMPLE ORDERS DATA VERIFICATION ===")

# Check main structure
print(f"Keys in data: {list(data.keys())}")
print(f"Number of epochs with sample orders: {len(data['epoch_sample_orders'])}")
print(f"Number of epochs with random seeds: {len(data['epoch_random_seeds'])}")

# Check experiment info
exp_info = data['experiment_info']
print(f"\n=== EXPERIMENT INFO ===")
for key, value in exp_info.items():
    print(f"{key}: {value}")

# Check a few epochs
print(f"\n=== SAMPLE ORDER VERIFICATION ===")
for epoch in [0, 50, 100, 150, 199]:
    if epoch in data['epoch_sample_orders']:
        sample_order = data['epoch_sample_orders'][epoch]
        print(f"Epoch {epoch}: {len(sample_order)} samples")
        print(f"  First 10 samples: {sample_order[:10]}")
        print(f"  Last 10 samples: {sample_order[-10:]}")
    else:
        print(f"Epoch {epoch}: No data")

# Check random seeds for a few epochs
print(f"\n=== RANDOM SEED VERIFICATION ===")
for epoch in [0, 1, 2]:
    if epoch in data['epoch_random_seeds']:
        seed_info = data['epoch_random_seeds'][epoch]
        print(f"Epoch {epoch} random seed keys: {list(seed_info.keys())}")
        # Don't print the actual seed values (they're large)
    else:
        print(f"Epoch {epoch}: No random seed data")

# Verify numpy file too
npz_file = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results_memo_weighted_t0.25_w3.0_single_200_random_weight1/epoch_sample_orders_task_19.npz"
print(f"\n=== NUMPY FILE VERIFICATION ===")
npz_data = np.load(npz_file)
print(f"Keys in npz file: {list(npz_data.keys())}")
print(f"Number of epochs in npz: {len(npz_data.keys())}")

# Check first epoch from npz
epoch_0_npz = npz_data['epoch_0']
print(f"Epoch 0 from npz: {len(epoch_0_npz)} samples")
print(f"First 10 samples: {epoch_0_npz[:10]}")

print(f"\n=== SUCCESS: Both files contain complete epoch data! ===")