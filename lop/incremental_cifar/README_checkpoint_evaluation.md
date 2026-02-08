# Checkpoint Evaluation Script

This script evaluates trained model checkpoints and calculates test accuracy on CIFAR-100 data.

## Usage

### Evaluate a single checkpoint file:
```bash
python evaluate_checkpoints.py \
    --checkpoint /path/to/checkpoint.pt \
    --data-path /path/to/cifar/data \
    --output-csv results.csv
```

### Evaluate all checkpoints in a results directory:
```bash
python evaluate_checkpoints.py \
    --results-dir /path/to/results/directory \
    --data-path /path/to/cifar/data
```

### Example with your checkpoint:
```bash
python evaluate_checkpoints.py \
    --checkpoint results_old/base_deep_learning_system/model_parameters/index-0_epoch-4000.pt \
    --data-path data/
```

## Arguments

- `--results-dir`: Directory containing `model_parameters/` folder with checkpoint files
- `--checkpoint`: Evaluate a single checkpoint file
- `--data-path`: Path to CIFAR data directory (optional, defaults to `./data/`)
- `--output-csv`: Output CSV file path (optional, auto-generated if not specified)
- `--device`: Device to use (cuda:0, cpu, etc.) - auto-detected if not specified
- `--num-classes`: Number of classes to evaluate on (default: all 100 classes)

## Output

The script creates a CSV file with the following columns:
- `checkpoint_name`: Name of the checkpoint file
- `checkpoint_path`: Full path to checkpoint
- `test_accuracy`: Test accuracy (0.0 to 1.0)
- `test_loss`: Test loss value
- `num_classes_evaluated`: Number of classes used in evaluation
- `total_samples`: Total number of test samples
- `correct_predictions`: Number of correct predictions

## Requirements

The script uses the same dependencies as the main experiment:
- PyTorch
- torchvision 
- numpy
- mlproj_manager (from your project)
- lop.nets (ResNet implementation)
