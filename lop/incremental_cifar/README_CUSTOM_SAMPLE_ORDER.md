# Custom Predetermined Sample Order

This guide explains how to use custom CSV files for predetermined sample ordering in the incremental CIFAR experiments.

## Overview

The system now supports two ways to specify predetermined sample order:

1. **Default mode**: Use built-in `sample_class_map_ascending.csv` or `sample_class_map_descending.csv`
2. **Custom mode**: Use your own CSV file with custom sample ordering

## Usage

### Method 1: Default Files (Original Behavior)

```bash
# In run_incremental_cifar_memo_ordered_single.sh
PREDETERMINED_SAMPLE_ORDER="ascending"  # or "descending"
PREDETERMINED_SAMPLE_CSV_PATH=""        # Leave empty for default
```

### Method 2: Custom CSV File

```bash
# In run_incremental_cifar_memo_ordered_single.sh
PREDETERMINED_SAMPLE_ORDER="ascending"  # Still required (used for directory naming)
PREDETERMINED_SAMPLE_CSV_PATH="/path/to/your/custom_sample_order.csv"
```

## CSV File Format

Your custom CSV file must have these columns:

```csv
sample_id,memorization_score,class_label
0,0.123456,42
15,0.234567,85
27,0.345678,12
...
```

### Required Columns:
- `sample_id`: Integer index of the sample in the CIFAR-100 dataset
- `memorization_score`: Float value representing memorization difficulty
- `class_label`: Integer class label (0-99 for CIFAR-100)

### Example Custom Ordering Strategies:

1. **Random shuffling**: Randomize sample order
2. **Class-balanced**: Ensure equal representation from each class
3. **Difficulty gradual**: Gradually increase difficulty over training
4. **Mixed strategies**: Combine multiple ordering principles

## Example Usage

```bash
# Create your custom CSV file
echo "sample_id,memorization_score,class_label" > my_custom_order.csv
echo "42,0.1,5" >> my_custom_order.csv
echo "123,0.2,7" >> my_custom_order.csv
# ... add 49,998 more samples

# Configure the script
PREDETERMINED_SAMPLE_ORDER="ascending"  # For directory naming
PREDETERMINED_SAMPLE_CSV_PATH="/path/to/my_custom_order.csv"

# Run the experiment
./run_incremental_cifar_memo_ordered_single.sh
```

## Output Directory Naming

When using a custom CSV file, the results directory will still use the `PREDETERMINED_SAMPLE_ORDER` value for naming:

- Default: `results_predetermined_sample_ascending_random_scratch_fixed_200_single_task_19_200epochs`
- Custom: Same naming pattern, but uses your custom file internally

## Validation

The system will validate your CSV file:
- Must have required columns
- Must have exactly 50,000 samples (CIFAR-100 training set size)
- Sample IDs must be valid indices (0-49,999)
- Will print first 10 samples and memo score range for verification

## Error Handling

If your custom CSV file has issues:
- Missing file: Error message with file path
- Wrong format: Column validation error  
- Invalid sample IDs: Index validation error
- Wrong number of samples: Count validation error