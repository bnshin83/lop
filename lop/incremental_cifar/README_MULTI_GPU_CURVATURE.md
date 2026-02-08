# Multi-GPU Curvature Analysis

This document describes the new multi-GPU curvature analysis system that processes **ALL samples** for **ALL epochs** using efficient cluster computing.

## 🎯 Overview

The multi-GPU curvature analysis system consists of two main components:

1. **`multi_gpu_curvature_analysis.py`** - Main computation script with distributed processing
2. **`run_multi_gpu_curvature.sh`** - SLURM job script for cluster execution

## ✨ Key Features

### **No Sample Limitations**
- Processes **ALL 50,000 CIFAR-100 samples** (no max_samples limit)
- Complete coverage across all training epochs

### **Multi-GPU Distributed Processing**
- Uses PyTorch Distributed Data Parallel (DDP)
- Supports multi-node cluster execution
- Efficient memory management and GPU utilization

### **Optimized for Cluster**
- SLURM-compatible job submission
- Automatic checkpointing and resume capability
- Comprehensive logging and error handling

### **Proper Path Configuration**
- Uses `results2/base_deep_learning_system` structure
- Saves to `per_image_curvature_full` directory
- Compatible with existing model parameters

### **🎯 CRITICAL: Precise Class Partitioning**
- **Follows incremental_cifar_experiment.py class ordering exactly**
- Loads class order from `class_order/index-0.npy`
- Respects incremental learning logic: 5 classes → +5 every 200 epochs
- Only processes classes that were active during each epoch's training
- Ensures curvature analysis matches the original incremental learning setup

## 📁 Directory Structure

```
results2/base_deep_learning_system/
├── model_parameters/                    # Input: trained model weights
│   ├── index-0_epoch-0.pt
│   ├── index-0_epoch-200.pt
│   └── ...
└── per_image_curvature_full/           # Output: curvature analysis
    ├── epoch_0000/
    │   ├── per_image_curvature.npy      # Curvature values (50,000 samples)
    │   ├── per_image_directional_eig.npy # Directional eigenvalues
    │   ├── per_image_labels.npy         # Class labels
    │   └── metadata.json               # Analysis metadata
    ├── epoch_0200/
    └── ...
```

## 🚀 Usage

### **Option 1: Submit SLURM Job (Recommended)**

```bash
# Submit to cluster queue
sbatch run_multi_gpu_curvature.sh

# Monitor job status
squeue -u $USER

# View progress
tail -f log/[JOB_ID]_multi_gpu_curvature.out
```

### **Option 2: Interactive Multi-GPU**

```bash
# Request interactive session with multiple GPUs
salloc --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --time=4:00:00

# Run with torchrun
torchrun --nproc_per_node=4 multi_gpu_curvature_analysis.py \
    --start_epoch 0 --end_epoch 4000 --step 200
```

### **Option 3: Custom Epoch Range**

```bash
# Process specific epochs
python multi_gpu_curvature_analysis.py \
    --epochs "0,500,1000,2000,4000" \
    --data_path /path/to/cifar-100-python \
    --model_dir /path/to/model_parameters \
    --save_base_dir /path/to/output
```

## ⚙️ Configuration

### **Default Parameters**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `start_epoch` | 0 | Starting epoch number |
| `end_epoch` | 4000 | Ending epoch number |
| `step` | 200 | Step size between epochs |
| `data_path` | `data/cifar-100-python` | CIFAR-100 dataset path |
| `model_dir` | `results2/.../model_parameters` | Model weights directory |
| `save_base_dir` | `results2/.../per_image_curvature_full` | Output directory |
| `experiment_index` | 0 | Experiment index for model files |

### **SLURM Resource Configuration**

```bash
#SBATCH --nodes=2              # 2 compute nodes
#SBATCH --ntasks-per-node=4    # 4 tasks per node
#SBATCH --gpus-per-node=4      # 4 GPUs per node
#SBATCH --time=7-00:00:00      # 7 days max runtime
```

**Total Resources**: 8 GPUs across 2 nodes

## 📊 Performance Estimates

### **Processing Scale**
- **Samples per epoch**: 50,000
- **Epochs to process**: 21 (0, 200, 400, ..., 4000)
- **Total samples**: 1,050,000
- **Output size**: ~15-20 GB

### **Timing Estimates**
- **Per epoch (8 GPUs)**: ~10-15 minutes
- **Total runtime**: ~5-7 hours
- **Throughput**: ~1,500-2,000 samples/second

## 🎯 Class Partitioning Logic

### **Incremental Learning Class Progression**

The curvature analysis **precisely follows** the incremental learning setup:

| Epoch Range | Task | Active Classes | Description |
|-------------|------|----------------|-------------|
| 0-199 | 0 | 5 classes | Initial training phase |
| 200-399 | 1 | 10 classes | +5 new classes added |
| 400-599 | 2 | 15 classes | +5 new classes added |
| ... | ... | ... | ... |
| 3800-4000 | 19 | 100 classes | All classes active |

### **Class Order Implementation**
```python
# Load class order from incremental experiment
class_order = np.load("class_order/index-0.npy")  # e.g., [8, 44, 23, 58, 49, ...]

# For each epoch, compute active classes
def compute_active_classes_for_epoch(epoch):
    task_number = epoch // 200
    current_num_classes = 5 + (task_number * 5)
    return class_order[:current_num_classes]

# Examples:
# Epoch 0:    classes [8, 44, 23, 58, 49]
# Epoch 200:  classes [8, 44, 23, 58, 49, 80, 64, 86, 30, 92]
# Epoch 1000: classes [8, 44, 23, 58, 49, ..., first_30_from_order]
```

### **Why This Matters**
- **Consistency**: Curvature computed only on data the model was trained on at each epoch
- **Comparability**: Results directly comparable to incremental learning analysis
- **Accuracy**: Avoids analyzing classes the model hadn't seen yet during training

## 🔧 Technical Details

### **Distributed Processing**
- Uses NCCL backend for GPU communication
- DistributedSampler ensures no data duplication
- Automatic result gathering from all processes

### **Memory Management**
- Batch size: 32 (adjustable based on GPU memory)
- Periodic GPU cache clearing
- Efficient tensor operations

### **Error Handling**
- Automatic checkpoint detection
- Resume capability for interrupted jobs
- Comprehensive error logging

## 📈 Output Format

### **Per-Epoch Files**

Each epoch directory contains:

1. **`per_image_curvature.npy`**
   - Shape: `(50000,)` 
   - Curvature values for each sample

2. **`per_image_directional_eig.npy`**
   - Shape: `(50000,)`
   - Directional eigenvalues for each sample

3. **`per_image_labels.npy`**
   - Shape: `(50000,)`
   - CIFAR-100 class labels (0-99)

4. **`metadata.json`**
   - Analysis statistics and configuration

### **Sample Metadata**

```json
{
  "epoch": 1000,
  "total_samples": 50000,
  "num_gpus": 8,
  "compute_time_seconds": 847.3,
  "allowed_classes": [0, 1, 2, ..., 99],
  "curvature_stats": {
    "mean": 0.0234,
    "std": 0.0156,
    "min": 0.0001,
    "max": 0.2341
  }
}
```

## 🚨 Prerequisites

### **Required Modules**
```bash
module load cuda/12.1
module load nccl/2.18.1-cuda12.1
```

### **Conda Environment**
```bash
conda activate /scratch/gautschi/shin283/conda_envs/lop
```

### **Required Files**
- Model parameters: `results2/base_deep_learning_system/model_parameters/index-0_epoch-*.pt`
- CIFAR-100 dataset: `data/cifar-100-python/`

## ✅ Verification

### **Test Class Partitioning Logic**
```bash
# Verify class partitioning follows incremental learning exactly
python verify_class_partitioning.py
```

Expected output:
```
📋 Loaded class order: [ 8 44 23 58 49 80 64 86 30 92 38  5 25 52  6 74 14 40 37 45]...
🧪 Testing Class Partitioning Logic:
Epoch  Task #Classes  Active Classes
======================================================================
0      0    5         [8, 44, 23, 58, 49]
200    1    10        [8, 44, 23, 58, 49, 80, 64, 86, 30, 92]
400    2    15        [8, 44, 23, 58, 49, 80, 64, 86, 30, 92, ...]
...
✅ Class increment logic is correct!
```

## 📋 Monitoring & Debugging

### **Check Progress**
```bash
# Count completed epochs
ls results2/base_deep_learning_system/per_image_curvature_full/*/per_image_curvature.npy | wc -l

# View latest log
tail -f log/[JOB_ID]_multi_gpu_curvature.out

# Check GPU usage
nvidia-smi
```

### **Common Issues**

1. **CUDA OOM**: Reduce batch size in script
2. **NCCL errors**: Check network configuration
3. **Missing models**: Verify model_parameters directory
4. **Disk space**: Ensure sufficient storage (~20GB)

## 🔄 Integration with Existing Pipeline

This multi-GPU curvature analysis integrates seamlessly with:

- **CSL/LT Analysis**: Use curvature data alongside progression cache
- **Visualization Scripts**: Plot curvature evolution across epochs  
- **Class-ordered Analysis**: Apply class ordering from `index-0.npy`

## 📞 Support

For issues or modifications:
1. Check SLURM logs in `log/` directory
2. Verify GPU availability with `nvidia-smi`
3. Ensure proper conda environment activation
4. Check disk space and file permissions

---

**Created**: August 2025  
**Purpose**: Full-scale curvature analysis for incremental CIFAR-100 experiments  
**Cluster**: Multi-GPU distributed processing optimized
