#!/usr/bin/env python3
"""
Quick test to debug predetermined weights issue
"""
import sys
import os

# Test the predetermined weights functionality with debugging
print("Testing predetermined weights with debugging...")

# Run just 1 epoch to see debug output
cmd = """python3.8 incremental_cifar_memo_ordered_experiment.py \
--config_file ./cfg/base_deep_learning_system.json \
--results_dir /tmp/test_predetermined_weights \
--use_predetermined_weights \
--predetermined_weights_csv_path sample_cifar100_high_infl_pairs_infl0.15_mem0.25_combined_with_classes_CORRECTED.csv \
--weight_dramaticity 5.0 \
--start_task 19 \
--max_tasks 1 \
--epochs_per_task 1 \
--scratch \
--class_order random \
--memo_order low_to_high \
--run_index 0 \
--gpu_id 0"""

print("Running command:")
print(cmd)
print("\n" + "="*80)

os.system(cmd)