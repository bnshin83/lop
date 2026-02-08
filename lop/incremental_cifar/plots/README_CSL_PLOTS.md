# CSL and Learning-Time Plotting Scripts

This directory contains plotting scripts specifically designed for analyzing Cumulative Sample Loss (CSL) and Learning-Time data from incremental CIFAR experiments.

## New Scripts

### 1. `plot_csl_lt_epochs.py`
**Main epoch-wise plotting script** following the structure of `plot_incremental_cifar_results_h.py`.

#### Supported Metrics:
- **`mean_loss_per_epoch`**: Average loss per sample across epochs
- **`mean_grad_per_epoch`**: Average gradient norm per sample across epochs  
- **`csl_progression`**: Cumulative Sample Loss progression across epochs
- **`lt_progression`**: Learning-Time progression across epochs

#### Usage:
```bash
python3 plots/plot_csl_lt_epochs.py \
    --experiments "base_deep_learning_system,head_resetting" \
    --metric "csl_progression" \
    --results_dir "./results/" \
    --max_samples 1000
```

#### Key Features:
- **Memory Efficient**: Use `--max_samples` to limit samples for faster processing
- **Error Bars**: Shows standard error across samples
- **Multiple Experiments**: Compare different algorithms on same plot
- **Flexible Metrics**: Support for various CSL/LT related metrics

### 2. `plot_csl_lt_analysis.py` 
**Comprehensive analysis script** for CSL/LT distributions and relationships.

#### Supported Plot Types:
- **`csl_distribution`**: Histogram of CSL values across experiments
- **`lt_distribution`**: Histogram of Learning-Time values  
- **`loss_per_epoch`**: Epoch-wise loss evolution
- **`summary`**: 4-panel comprehensive summary plot

#### Usage:
```bash
python3 plots/plot_csl_lt_analysis.py \
    --experiments "base_deep_learning_system" \
    --plot_type "summary" \
    --results_dir "./results/"
```

### 3. `test_csl_plots.py`
**Test script** that runs all plotting functions to verify functionality.

## Data Requirements

The scripts expect the following directory structure:
```
results/
├── base_deep_learning_system/
│   ├── csl/
│   │   ├── csl_cifar100_run1_noise_0.01.npy
│   │   ├── lt_cifar100_run1_noise_0.01.npy
│   │   └── metrics_cifar100_run1_noise_0.01.json
│   └── per_sample_losses_inc/
│       ├── loss_cifar100_epoch_0.npy
│       ├── loss_grad_cifar100_epoch_0.npy
│       └── ... (for each epoch)
└── other_experiments/
    └── ... (same structure)
```

## Generated Outputs

All plots are saved as SVG files in the `plots/` directory:
- `csl_progression_epochs.svg`
- `lt_progression_epochs.svg`
- `mean_loss_per_epoch_epochs.svg`
- `mean_grad_per_epoch_epochs.svg`

## Example Commands

### Compare CSL progression across multiple experiments:
```bash
python3 plots/plot_csl_lt_epochs.py \
    --experiments "base_deep_learning_system,retrained_network,head_resetting" \
    --metric "csl_progression" \
    --max_samples 5000
```

### Generate comprehensive summary:
```bash
python3 plots/plot_csl_lt_analysis.py \
    --experiments "base_deep_learning_system" \
    --plot_type "summary"
```

### Quick test of all functionality:
```bash
python3 plots/test_csl_plots.py
```

## Integration with Existing Workflow

These scripts complement the existing `plot_incremental_cifar_results_h.py` by:

1. **Using the same color scheme** for consistent visualization
2. **Following the same argument structure** for easy integration
3. **Supporting multiple experiments** for comparative analysis
4. **Providing similar output formats** (SVG files)

## Key Differences from Original Script

- **Per-sample focus**: Analyzes individual sample behavior rather than aggregate metrics
- **CSL-specific metrics**: Computes cumulative sample loss and learning-time progressions
- **Memory efficiency**: Supports sampling for large datasets
- **Epoch-wise analysis**: Tracks progression across training epochs

## Dependencies

- `matplotlib`
- `numpy`
- `pandas` (for `plot_csl_lt_analysis.py`)
- Standard Python libraries (`os`, `argparse`, `json`)

No external plotting libraries required - all functions are self-contained.

## Notes

- **Gradient files**: If gradient files are missing, the scripts will use loss files as proxies
- **Tau calculation**: Learning-time progression automatically computes tau thresholds
- **Sample indices**: Random sampling is used for memory efficiency when `max_samples` is specified
- **Error handling**: Scripts gracefully handle missing files and provide informative warnings
