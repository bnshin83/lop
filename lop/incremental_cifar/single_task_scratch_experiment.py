#!/usr/bin/env python3
"""
Single-task scratch mode experiment script.
Runs exactly ONE task from complete scratch, with no dependencies on other tasks.
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from incremental_cifar_memo_ordered_experiment import IncrementalCIFARMemoOrderedExperiment


def main():
    parser = argparse.ArgumentParser(description="Run a single task from scratch")
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to run (0-19)")
    parser.add_argument("--config", type=str, default="./cfg/base_deep_learning_system.json", help="Config file path")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to run for this task")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--no_ordering", action="store_true", help="Disable memo ordering (random shuffle)")
    parser.add_argument("--memo_order", type=str, default="low_to_high", help="Memorization ordering")
    
    args = parser.parse_args()
    
    # Validate task ID
    if not 0 <= args.task_id <= 19:
        raise ValueError(f"Task ID must be between 0-19, got {args.task_id}")
    
    # Use the provided results directory directly (SLURM script already creates task-specific dirs)
    task_results_dir = args.results_dir
    os.makedirs(task_results_dir, exist_ok=True)
    
    # Load config file
    import json
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set default data path if empty (similar to the original experiment script)
    if not config.get("data_path"):
        config["data_path"] = "/scratch/gautschi/shin283/loss-of-plasticity/lop/data"
    
    print(f"="*80)
    print(f"SINGLE TASK SCRATCH MODE EXPERIMENT")
    print(f"Task ID: {args.task_id}")
    print(f"GPU: {args.gpu_id}")
    print(f"Epochs: {args.epochs}")
    print(f"No ordering: {args.no_ordering}")
    print(f"Results dir: {task_results_dir}")
    print(f"="*80)
    
    # Create experiment instance configured for single task
    experiment = IncrementalCIFARMemoOrderedExperiment(
        exp_params=config,
        results_dir=task_results_dir,
        run_index=0,  # Always use run_index 0 for single tasks
        verbose=True,
        memo_order=args.memo_order,
        gpu_id=args.gpu_id,
        no_ordering=args.no_ordering,
        scratch=True,  # Always scratch mode
        epochs_per_task=args.epochs,
        class_order="sequential" if not args.no_ordering else "memo_low_to_high",  # Use sequential when sample ordering is active
        incremental_epochs=False,
        start_task=args.task_id,  # Start at the specific task
        max_tasks=1  # Run only 1 task
    )
    
    # Load the predetermined class order (same as memo-ordered experiment)
    full_class_order = experiment.load_class_order()
    
    # For single-task scratch mode, use the SAME logic as original experiment
    # but with sequential class order to avoid memo-based class reordering
    total_classes_for_task = (args.task_id + 1) * 5
    
    # Set up the class list for this specific task using SEQUENTIAL order (like original)
    experiment.current_num_classes = total_classes_for_task
    experiment.all_classes = np.arange(experiment.num_classes)  # Use full sequential [0,1,2,...,99] like original
    
    print(f"Task {args.task_id} will train on {total_classes_for_task} classes total")
    print(f"Classes: {experiment.all_classes}")
    task_specific_classes = experiment.all_classes  # Use all classes, not just the new 5
    
    print(f"Task {args.task_id} will train on predetermined classes {list(task_specific_classes)}")
    print(f"Full class list for partitioning: {list(experiment.all_classes)}")
    print(f"Current num classes: {experiment.current_num_classes}")
    
    # Run the experiment
    try:
        experiment.run()
        print(f"✅ Task {args.task_id} completed successfully!")
        
        # Save task metadata
        metadata = {
            "task_id": args.task_id,
            "classes_range": [start_class, end_class-1],
            "total_classes": end_class,
            "epochs": args.epochs,
            "gpu_id": args.gpu_id,
            "scratch_mode": True,
            "no_ordering": args.no_ordering,
            "config_file": args.config
        }
        
        metadata_path = os.path.join(task_results_dir, f"task_{args.task_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"❌ Task {args.task_id} failed with error: {e}")
        raise


if __name__ == "__main__":
    main()