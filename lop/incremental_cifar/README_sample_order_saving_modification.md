# Modifying Predetermined Mode to Save Sample Orders by Random Seeds

## Overview

This document outlines the modifications needed to save sample orders for all epochs in the predetermined mode, organized by random seeds and saved as NPZ/NPY files for efficient storage and analysis.

## Current Behavior vs Desired Behavior

### Current Behavior
- **Random mode** (`no_ordering=True`): Shuffles samples differently each epoch, but only saves the first epoch's sample order
- **Predetermined mode** (`predetermined_sample_order="ascending"`): Uses fixed sample order for all epochs, saves only once

### Desired Behavior
- Save sample orders for **all epochs** in both modes
- Organize saved orders by **random seed** for reproducibility
- Use **NPZ/NPY format** for efficient storage and fast loading
- Enable analysis of epoch-to-epoch sample order variation

## Required Modifications

### 1. Add Epoch-Level Sample Order Tracking

#### Location: `IncrementalCIFARMemoOrderedExperiment.__init__()`

Add new instance variables to track sample orders:

```python
# Add to constructor after existing sample order variables
self.sample_orders_by_epoch = {}  # Dict to store sample orders for each epoch
self.save_all_epoch_orders = True  # Flag to enable/disable saving all epochs
self.sample_orders_output_format = "npz"  # "npz", "npy", or "csv"
```

### 2. Modify Sample Order Extraction for All Epochs

#### Location: `get_data()` method, training data section

Replace the current single-epoch extraction with multi-epoch tracking:

```python
# Current code (lines ~1185-1188):
if not validation:
    actual_training_order = self._extract_actual_training_order(dataloader, ordered_indices, shuffle_data)
    self._actual_training_order = actual_training_order

# New code:
if not validation:
    if shuffle_data:
        # For shuffled data, we need to track the order for each epoch
        self._setup_epoch_order_tracking(dataloader, ordered_indices)
    else:
        # For fixed order, same order used for all epochs
        actual_training_order = self._extract_actual_training_order(dataloader, ordered_indices, shuffle_data)
        self._actual_training_order = actual_training_order
```

### 3. Add New Method: `_setup_epoch_order_tracking()`

```python
def _setup_epoch_order_tracking(self, dataloader, ordered_indices):
    """
    Set up tracking of sample orders for each epoch when shuffle=True.
    Pre-generates sample orders for all epochs using the experiment's random seed.
    """
    print(f"DEBUG: Setting up epoch-level sample order tracking for {self.epochs_per_task} epochs")
    
    # Store the current random state to restore later
    current_state = torch.get_rng_state()
    
    # Set the random seed to match the experiment
    torch.manual_seed(int(self.random_seed))
    
    self.sample_orders_by_epoch = {}
    
    for epoch in range(self.epochs_per_task):
        # Generate the same random order that DataLoader will use for this epoch
        # Note: This simulates the RandomSampler behavior
        shuffle_order = torch.randperm(len(ordered_indices))
        epoch_sample_order = ordered_indices[shuffle_order]
        
        self.sample_orders_by_epoch[epoch] = {
            'training_samples': epoch_sample_order.clone(),
            'random_seed': int(self.random_seed),
            'epoch': epoch,
            'shuffle_order': shuffle_order.clone()
        }
    
    # Set the first epoch's order for immediate use
    self._actual_training_order = self.sample_orders_by_epoch[0]['training_samples']
    
    # Restore the original random state
    torch.set_rng_state(current_state)
    
    print(f"DEBUG: Generated sample orders for {len(self.sample_orders_by_epoch)} epochs")
    print(f"DEBUG: First epoch order: {self._actual_training_order[:10].tolist()}")
```

### 4. Add Method: `_save_all_epoch_sample_orders()`

```python
def _save_all_epoch_sample_orders(self, validation_order, task_id):
    """
    Save sample orders for all epochs in NPZ/NPY format.
    
    :param validation_order: Validation sample indices
    :param task_id: Current task identifier
    """
    if not self.save_all_epoch_orders or not hasattr(self, 'sample_orders_by_epoch'):
        return
    
    print(f"DEBUG: Saving sample orders for all epochs...")
    
    # Prepare data structure for saving
    save_data = {
        'metadata': {
            'random_seed': int(self.random_seed),
            'task_id': task_id,
            'num_epochs': len(self.sample_orders_by_epoch),
            'training_samples_per_epoch': len(self._actual_training_order),
            'validation_samples': len(validation_order),
            'total_samples_per_epoch': len(self._actual_training_order) + len(validation_order),
            'shuffle_enabled': self.no_ordering,
            'sample_order_mode': 'random' if self.no_ordering else f'predetermined_{self.predetermined_sample_order}',
            'experiment_timestamp': time.time()
        }
    }
    
    # Add epoch-specific data
    for epoch, epoch_data in self.sample_orders_by_epoch.items():
        # Combine training + validation for complete 50k sample order
        complete_epoch_order = torch.cat([
            epoch_data['training_samples'],
            validation_order
        ])
        
        save_data[f'epoch_{epoch:03d}'] = {
            'sample_order': complete_epoch_order.numpy(),
            'training_samples': epoch_data['training_samples'].numpy(),
            'validation_samples': validation_order.numpy(),
            'shuffle_order': epoch_data['shuffle_order'].numpy() if 'shuffle_order' in epoch_data else None
        }
    
    # Generate filename
    timestamp = int(time.time())
    filename_base = f"sample_orders_task_{task_id}_seed_{int(self.random_seed)}_{timestamp}"
    
    if self.sample_orders_output_format == "npz":
        filepath = os.path.join(self.results_dir, f"{filename_base}.npz")
        np.savez_compressed(filepath, **save_data)
        print(f"Saved sample orders to: {filepath}")
        
    elif self.sample_orders_output_format == "npy":
        # Save each epoch separately as NPY files
        output_dir = os.path.join(self.results_dir, f"sample_orders_{filename_base}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(save_data['metadata'], f, indent=2)
        
        # Save each epoch
        for epoch in range(len(self.sample_orders_by_epoch)):
            epoch_path = os.path.join(output_dir, f"epoch_{epoch:03d}.npy")
            np.save(epoch_path, save_data[f'epoch_{epoch:03d}']['sample_order'])
        
        print(f"Saved sample orders to directory: {output_dir}")
    
    # Also create a summary CSV for quick inspection
    summary_data = []
    for epoch in range(len(self.sample_orders_by_epoch)):
        epoch_order = save_data[f'epoch_{epoch:03d}']['sample_order']
        summary_data.append({
            'epoch': epoch,
            'first_sample_id': int(epoch_order[0]),
            'last_training_sample_id': int(epoch_order[len(self._actual_training_order)-1]),
            'first_validation_sample_id': int(epoch_order[len(self._actual_training_order)]),
            'unique_samples': len(np.unique(epoch_order))
        })
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(self.results_dir, f"{filename_base}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved sample order summary to: {summary_path}")
```

### 5. Update the Complete Order Saving Logic

#### Location: `get_data()` method, validation data section

Replace the current single CSV saving with multi-epoch NPZ saving:

```python
# Current code (lines ~1194-1212):
if hasattr(self, '_actual_training_order'):
    # ... existing CSV saving code ...

# New code:
if hasattr(self, '_actual_training_order'):
    if self.save_all_epoch_orders and hasattr(self, 'sample_orders_by_epoch'):
        # Save all epoch orders in NPZ/NPY format
        self._save_all_epoch_sample_orders(self._actual_validation_order, self.current_task)
    else:
        # Fallback to single epoch CSV saving (existing behavior)
        complete_order = torch.cat([self._actual_training_order, self._actual_validation_order])
        # ... existing CSV saving code ...
```

### 6. Add Command Line Arguments

#### Location: Main execution section or argument parser

```python
# Add to argument parser
parser.add_argument("--save_all_epoch_orders", action="store_true", default=False,
                   help="Save sample orders for all epochs (not just first epoch)")
parser.add_argument("--sample_orders_format", type=str, default="npz", 
                   choices=["npz", "npy", "csv"],
                   help="Format for saving sample orders")
```

### 7. Configuration Updates

#### Location: Experiment scripts

Add configuration options to shell scripts:

```bash
# Add to script variables
SAVE_ALL_EPOCH_ORDERS="true"  # Set to "true" to enable all-epoch saving
SAMPLE_ORDERS_FORMAT="npz"    # Options: "npz", "npy", "csv"

# Add to command arguments
if [ "$SAVE_ALL_EPOCH_ORDERS" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --save_all_epoch_orders"
fi
CMD_ARGS="$CMD_ARGS --sample_orders_format $SAMPLE_ORDERS_FORMAT"
```

## File Structure After Modifications

```
results_random_single_200/
├── sample_orders_task_19_seed_37542_1693401234.npz          # All epochs in one file
├── sample_orders_task_19_seed_37542_1693401234_summary.csv  # Quick summary
├── actual_sample_order_task_19_epoch_0.csv                  # Legacy format (first epoch only)
└── ...existing files...

# OR if using NPY format:
results_random_single_200/
├── sample_orders_task_19_seed_37542_1693401234/
│   ├── metadata.json
│   ├── epoch_000.npy
│   ├── epoch_001.npy
│   ├── ...
│   └── epoch_199.npy
└── ...existing files...
```

## Usage Examples

### Loading Sample Orders for Analysis

```python
import numpy as np

# Load NPZ file
data = np.load('sample_orders_task_19_seed_37542_1693401234.npz', allow_pickle=True)

# Access metadata
metadata = data['metadata'].item()
print(f"Random seed: {metadata['random_seed']}")
print(f"Number of epochs: {metadata['num_epochs']}")

# Access specific epoch
epoch_5_order = data['epoch_005']['sample_order']
print(f"Epoch 5 sample order: {epoch_5_order[:10]}")

# Compare orders across epochs
epoch_0_order = data['epoch_000']['sample_order']
epoch_1_order = data['epoch_001']['sample_order']
orders_identical = np.array_equal(epoch_0_order, epoch_1_order)
print(f"Epoch 0 and 1 orders identical: {orders_identical}")
```

### Analyzing Sample Order Variation

```python
# Calculate order variation across epochs
def analyze_order_variation(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    metadata = data['metadata'].item()
    
    variations = []
    for epoch in range(1, metadata['num_epochs']):
        prev_order = data[f'epoch_{epoch-1:03d}']['sample_order']
        curr_order = data[f'epoch_{epoch:03d}']['sample_order']
        
        # Calculate position changes
        changes = np.sum(prev_order != curr_order)
        variations.append(changes)
    
    return variations
```

## Benefits of This Approach

1. **Efficiency**: NPZ format is compressed and fast to load
2. **Completeness**: Captures sample orders for all epochs, not just first
3. **Reproducibility**: Orders organized by random seed for exact replication
4. **Flexibility**: Supports both fixed and shuffled sample order modes
5. **Analysis-Friendly**: Easy to load and analyze epoch-to-epoch variation
6. **Backward Compatible**: Existing CSV format still generated for compatibility

## Testing the Modifications

1. Run experiments with `--save_all_epoch_orders` flag
2. Verify NPZ files are created with correct structure
3. Load NPZ files and verify sample orders match expected patterns
4. Compare random vs predetermined mode outputs
5. Test reproducibility by running same seed multiple times

## Performance Considerations

- NPZ files will be larger than single CSV files (~200 epochs × 50k samples)
- Consider using `np.savez_compressed()` for better compression
- For very large experiments, consider saving only selected epochs or using NPY format with separate files per epoch