# SLURM Script Generation for Custom Class Orders

## Overview

The `generate_slurm_script.sh` script creates properly named SLURM job scripts that ensure logs and results directories reflect the specific class ordering strategy being used.

## Usage

```bash
./generate_slurm_script.sh <strategy_name> [experiment_index] [task_epochs]
```

### Parameters

- `strategy_name`: Required. Name of the class ordering strategy (must match a CSV file)
- `experiment_index`: Optional. Experiment index number (default: 0)
- `task_epochs`: Optional. Number of epochs per task (default: 200)

### Available Strategies

The script looks for CSV files in the `custom_class_order/` directory:

- `mountain` - Mountain-shaped difficulty curve
- `sine` - Sine wave pattern in memorization scores  
- `highest_first_gradual` - Start with highest, alternate high/low
- `lowest_first_gradual` - Start with lowest, alternate low/high
- `highest_middle_gradual` - Highest score in middle, gradual buildup
- `low_to_high` - Traditional curriculum learning
- `high_to_low` - Anti-curriculum learning

## Examples

```bash
# Generate script for mountain strategy with default parameters
./generate_slurm_script.sh mountain

# Generate script for sine strategy with specific parameters
./generate_slurm_script.sh sine 1 400

# Generate script for curriculum learning approach
./generate_slurm_script.sh low_to_high 0 200
```

## Output

The script generates a SLURM file named: `run_incremental_cifar_<strategy>_idx<index>_ep<epochs>.sh`

### Generated Script Features

- **Job name**: `inc_cifar_<strategy>_idx<index>_ep<epochs>`
- **Log files**: `<jobid>_inc_cifar_<strategy>_idx<index>_ep<epochs>.{out,err}`
- **Results directory**: `results_<strategy>_idx<index>_ep<epochs>/`
- **CSV file path**: Automatically set to match the strategy

### Example Output

```bash
./generate_slurm_script.sh mountain 0 200
```

Creates: `run_incremental_cifar_mountain_idx0_ep200.sh` with:
- Job name: `inc_cifar_mountain_idx0_ep200`
- Logs: `*_inc_cifar_mountain_idx0_ep200.{out,err}`
- Results: `results_mountain_idx0_ep200/`

## Submitting Jobs

After generation, submit the job:

```bash
sbatch run_incremental_cifar_mountain_idx0_ep200.sh
```

## Error Handling

The script will error if:
- No strategy name is provided
- The corresponding CSV file doesn't exist in `custom_class_order/`

To generate missing CSV files, use:
```bash
cd custom_class_order/
python3.8 generate_custom_class_order.py
```