# CSL and Curvature Calculation Framework

A generalized framework for computing Cumulative Sample Loss (CSL) and curvature metrics across different datasets and algorithms.

## Overview

This framework provides tools to analyze neural network training dynamics through:
- **CSL (Cumulative Sample Loss)**: Measures per-sample difficulty by summing losses across training epochs
- **Learning-Time**: Quantifies how long samples take to learn using gradient-based metrics
- **Curvature Analysis**: Examines loss landscape geometry on a per-sample basis

## ⚠️ Critical Design Philosophy: Observational vs Interventional Analysis

**Key Finding**: Per-sample interventions (like fixed high-memorization sample ordering) can actually **harm plasticity** by forcing models to overfit to specific sample sequences, while random shuffling preserves plasticity through diverse gradient signals.

**Framework Approach**: 
- **✅ RECOMMENDED**: Use per-sample metrics for **measurement and analysis only**
- **❌ AVOID**: Changing training procedures based on per-sample rankings
- **✅ MAINTAIN**: Random sample shuffling during training for optimal plasticity preservation
- **✅ ANALYZE**: Post-hoc correlations and patterns in per-sample behavior

## CSL Calculation Workflow

### Step 1: Per-Sample Loss Extraction ([score_cifar100_loss.py](cci:7://file:///scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/score_cifar100_loss.py:0:0-0:0))

**What it does:**
- Loads trained model checkpoints for each epoch
- Computes **per-sample cross-entropy loss** for every training sample
- Computes **per-sample gradient norms** (optional but recommended)
- Includes **weight decay regularization** in the loss calculation to match training conditions

**Key implementation details:**
```python
# Per-sample loss with weight decay (lines 91-131)
def get_loss_and_grad_with_weight_decay(model, criterion, images, targets, weight_decay=1e-4):
    outputs = model(images)
    ce_losses = criterion(outputs, targets)  # Cross-entropy per sample
    
    # Add L2 regularization to match training loss
    l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
    total_losses = ce_losses + weight_decay * l2_reg
    
    # Compute gradient norms per sample
    # ... (individual backward passes for each sample)
```

**Outputs per epoch:**
- `loss_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise}.npy` - Per-sample losses (50,000 values for CIFAR-100)
- `loss_grad_cifar100_run{run_idx}_epoch_{epoch}_noise_{noise}.npy` - Per-sample gradient norms

### Step 2: CSL Aggregation ([compute_csl.py](cci:7://file:///scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/compute_csl.py:0:0-0:0))

**What it does:**
- Loads all per-epoch loss files
- Aggregates them into final CSL and learning-time metrics

**CSL Calculation (lines 112-114):**
```python
def compute_csl(losses: np.ndarray) -> np.ndarray:
    """CSL per-sample as sum of losses across epochs. losses shape: (E, N)."""
    return losses.sum(axis=0)  # Sum across epochs for each sample
```

**Learning-Time Calculation (lines 117-131):**
```python
def compute_learning_time_from_grads(loss_grads: np.ndarray):
    # Running average of gradients over epochs
    e_count = np.arange(1, loss_grads.shape[0] + 1)[:, None]
    learning_condition = np.cumsum(loss_grads, axis=0) / e_count
    
    # Tau threshold (global mean)
    tau = float(learning_condition.mean())
    
    # Learning-time: 1 - fraction of epochs below tau
    below = learning_condition < tau
    lt = 1.0 - below.mean(axis=0)
```

## Key Insights

1. **CSL is simply the sum** of per-sample losses across all training epochs
2. **Learning-time** measures how often a sample's gradient is above the global average (tau)
3. **Weight decay is included** to match actual training conditions
4. **Missing gradient files** fall back to using loss values as proxies
5. **Per-sample computation** allows identifying which specific samples are hard to learn

## File Naming Conventions

The scripts support two naming schemes:
- **Gautschi format**: `loss_{dataset}_epoch_{epoch}.npy`
- **Gilbreth format**: `loss_{dataset}_run{run_idx}_epoch_{epoch}_noise_{noise}.npy`

## Final Outputs

- `csl_{dataset}_run{run_idx}_noise_{noise}.npy` - CSL per sample
- `lt_{dataset}_run{run_idx}_noise_{noise}.npy` - Learning-time per sample  
- `metrics_{dataset}_run{run_idx}_noise_{noise}.json` - Metadata including tau value

This CSL metric is then used to identify samples that accumulate high loss over training, which correlates with memorization difficulty and can inform sample ordering strategies for improved learning.

## Curvature Analysis Workflow

### Per-Sample Curvature Calculation

**Purpose**: Analyze loss landscape geometry around individual samples to understand training dynamics

**Core Script**:
- `per_image_curve.py` - Per-sample curvature computation

### Mathematical Foundation

**Curvature Estimation using Finite Differences**:
```python
def compute_curvature_per_image(net, data_loader, h=1e-3, niter=10, temp=1.0):
    """
    Compute loss landscape curvature for each sample using finite difference approximation.
    
    Method:
    1. For each sample x, generate random perturbation v with elements ±1
    2. Scale perturbation: v = h * v (where h is step size, typically 1e-3)
    3. Compute loss difference: L(x + v) - L(x)
    4. Compute gradient of loss difference w.r.t. input: ∇_x[L(x + v) - L(x)]
    5. Estimate curvature metrics from gradient
    
    Args:
        net: Neural network model
        data_loader: DataLoader with samples
        h: Step size for finite differences (default: 1e-3)
        niter: Number of random directions to average (default: 10)
        temp: Temperature scaling for softmax (default: 1.0)
    """
```

### Key Implementation Details

**1. Random Direction Generation**:
```python
# Generate random direction vector with elements ±1
v = torch.randint_like(batch_data, high=2, device=device) * 2 - 1  # Values: -1 or +1
v = h * v  # Scale by step size
```

**2. Loss Difference Computation**:
```python
# Forward pass with perturbed and original inputs
outputs_pos = net(batch_data + v)  # Perturbed predictions
outputs_orig = net(batch_data)     # Original predictions

# Compute losses
loss_pos = criterion(outputs_pos / temp, batch_labels)
loss_orig = criterion(outputs_orig / temp, batch_labels)

# Gradient of loss difference
grad_diff = torch.autograd.grad(loss_pos - loss_orig, batch_data, create_graph=False)[0]
```

**3. Curvature Metrics**:
```python
# Overall curvature (gradient norm)
curvature = grad_diff.reshape(num_samples, -1).norm(dim=1)

# Directional eigenvalue (directional second derivative)
directional_eig = (v.reshape(num_samples, -1) * grad_diff.reshape(num_samples, -1)).sum(dim=1)
```

### Usage Example

**Basic curvature computation**:
```python
# Load trained model and data
net = build_resnet18(num_classes=100)
net.load_state_dict(torch.load(checkpoint_path))
data_loader = load_cifar_data(data_path, train=True)

# Compute curvature metrics
curvature, directional_eig, labels = compute_curvature_per_image(
    net=net,
    data_loader=data_loader,
    h=1e-3,      # Step size for finite differences
    niter=10,    # Number of random directions to average
    temp=1.0,    # Temperature scaling
    save_dir="./curvature_results"
)
```

### Curvature Outputs

**Per-Sample Files**:
- `per_image_curvature.npy` - Overall curvature values (gradient norms)
- `per_image_directional_eig.npy` - Directional eigenvalues
- `per_image_labels.npy` - Corresponding class labels

**Aggregated Results**:
- `curvature_results_{dataset}_{epoch}.csv` - Per-sample curvature metrics with metadata
- `curvature_summary_{dataset}.json` - Statistical summaries and parameters

### Interpretation of Curvature Metrics

**1. Overall Curvature (Gradient Norm)**:
- **High values**: Sample lies in sharp region of loss landscape
- **Low values**: Sample lies in flat region of loss landscape
- **Implications**: High curvature samples may cause training instability

**2. Directional Eigenvalues**:
- **Positive values**: Loss increases in the random direction (convex-like behavior)
- **Negative values**: Loss decreases in the random direction (concave-like behavior)
- **Magnitude**: Indicates strength of curvature in that specific direction

### Generalization for Different Datasets

**Dataset-Specific Adaptations**:
```python
# CIFAR-10/100 normalization
transform = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

# ImageNet normalization
transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Custom dataset - adjust accordingly
transform = transforms.Normalize(dataset_mean, dataset_std)
```

**Model Architecture Adaptations**:
- Update model loading to match training architecture
- Adjust checkpoint loading for different model types
- Handle different output dimensions and loss functions

### Enhanced Sample Tracking Features

### Real-Time Sample Metadata Tracking

**Purpose**: Track comprehensive sample information during training for detailed analysis

**Implementation**: Enhanced `incremental_cifar_memo_ordered_experiment.py` with built-in tracking

**Tracked Metadata**:
```python
self.sample_tracking = {
    'sample_ids': [],      # Original dataset sample IDs
    'memo_scores': [],     # Memorization scores for each sample
    'labels': [],          # True labels (raw or one-hot encoded)
    'classes': [],         # Class IDs (converted from labels)
    'epoch_nums': [],      # Which epoch each sample was seen
    'batch_nums': [],      # Which batch within epoch
    'task_nums': []        # Which task each sample belongs to
}
```

### Key Tracking Methods

**1. Automatic Metadata Collection**:
```python
def _track_sample_metadata(self, sample_ids, images, labels, epoch_num, batch_num):
    """Track comprehensive sample metadata during training."""
    # Convert labels to class IDs if one-hot encoded
    if labels.dim() > 1 and labels.shape[1] > 1:
        class_ids = torch.argmax(labels, dim=1).cpu().tolist()
    else:
        class_ids = labels.cpu().tolist()
    
    # Get memo scores from loaded dictionaries
    memo_scores = []
    for sample_id in sample_ids:
        if sample_id in self.sample_memo_scores:
            memo_scores.append(self.sample_memo_scores[sample_id])
        else:
            memo_scores.append(0.0)  # Default if not found
```

**2. Comprehensive Data Export**:
```python
def save_sample_tracking_data(self, filename_suffix=""):
    """Save all tracked sample metadata to CSV."""
    tracking_df = pd.DataFrame({
        'sample_id': self.sample_tracking['sample_ids'],
        'memo_score': self.sample_tracking['memo_scores'],
        'true_label': self.sample_tracking['labels'],
        'class_id': self.sample_tracking['classes'],
        'epoch_num': self.sample_tracking['epoch_nums'],
        'batch_num': self.sample_tracking['batch_nums'],
        'task_num': self.sample_tracking['task_nums']
    })
```

### Usage in Experiments

**Enable Tracking**:
```python
# During experiment initialization
experiment = IncrementalCIFARMemoOrderedExperiment(
    exp_params=config,
    results_dir="./results/tracked_experiment",
    run_index=0,
    save_epoch_orders=True  # Enable comprehensive tracking
)

# Run experiment (tracking happens automatically)
experiment.run()

# Save tracking data at the end
experiment.save_sample_tracking_data("_final")
```

**Output CSV Format**:
```csv
sample_id,memo_score,true_label,class_id,epoch_num,batch_num,task_num
1234,0.7543,[0,0,1,0,...],2,0,5,0
5678,0.2156,[0,1,0,0,...],1,0,5,0
9012,0.8901,[1,0,0,0,...],0,1,12,0
```

### Integration with CSL Analysis

**Enhanced Analysis Pipeline**:
1. **Train models** with incremental learning + sample tracking
2. **Extract per-sample losses** using CSL pipeline
3. **Compute curvature metrics** for same samples
4. **Correlate all metrics** using sample IDs:
   - CSL values per sample
   - Curvature metrics per sample
   - Memorization scores per sample
   - Training dynamics (epoch/batch seen)
   - Task progression information

**Advanced Analysis Capabilities**:
```python
# Load all data sources using sample_id as key
csl_data = np.load("csl_cifar100_run1_noise_0.01.npy")
curvature_data = np.load("per_image_curvature.npy")
tracking_data = pd.read_csv("sample_tracking_data_final.csv")

# Merge by sample_id for comprehensive analysis
merged_analysis = tracking_data.merge(
    pd.DataFrame({'sample_id': range(len(csl_data)), 'csl': csl_data}),
    on='sample_id'
).merge(
    pd.DataFrame({'sample_id': range(len(curvature_data)), 'curvature': curvature_data}),
    on='sample_id'
)

# Now analyze relationships between:
# - memo_score vs csl vs curvature
# - Training order effects (epoch_num, batch_num)
# - Task-specific patterns (task_num)
# - Class-specific behaviors (class_id)
```

**Multi-Dimensional Sample Classification**:
- **High CSL + High Curvature + High Memo**: Most problematic samples
- **High CSL + Low Curvature + High Memo**: Difficult but stable samples  
- **Low CSL + High Curvature + Low Memo**: Easy but potentially disruptive samples
- **Low CSL + Low Curvature + Low Memo**: Ideal samples for early training
- **Training Order Effects**: Early vs late presentation impact
- **Task Progression**: How sample difficulty changes across tasks

### Generalization for New Datasets

**Adaptation Requirements**:
1. **Update sample ID mapping**: Ensure consistent sample identification across dataset
2. **Modify memo score loading**: Adapt to dataset-specific memorization metrics
3. **Adjust label handling**: Handle different label formats (categorical, one-hot, multi-label)
4. **Extend class mapping**: Support different number of classes and class hierarchies

**Example for ImageNet**:
```python
# Modify _track_sample_metadata for ImageNet
def _track_sample_metadata_imagenet(self, sample_ids, images, labels, epoch_num, batch_num):
    # ImageNet has 1000 classes, different normalization
    class_ids = labels.cpu().tolist()  # Already class indices
    
    # Load ImageNet-specific memorization scores
    memo_scores = []
    for sample_id in sample_ids:
        memo_scores.append(self.imagenet_memo_scores.get(sample_id, 0.0))
    
    # Track with ImageNet-specific metadata
    self.sample_tracking['sample_ids'].extend(sample_ids)
    self.sample_tracking['memo_scores'].extend(memo_scores)
    # ... rest of tracking logic
```

This enhanced framework provides complete traceability of sample behavior throughout training, enabling sophisticated analysis of learning dynamics, memorization patterns, and the effectiveness of different sample ordering strategies.

## Per-Sample Dormant Unit Proportion Calculation

### Multi-GPU Dormant Analysis Pipeline

**Purpose**: Calculate per-sample dormant unit proportions across all network layers for comprehensive plasticity analysis

**Implementation**: `multi_gpu_dormant_analysis.py` - distributed processing for efficient cluster computation

**Key Features**:
- **Per-sample granularity**: Individual dormant proportions (not batch-averaged)
- **Multi-GPU distributed processing**: Scalable across cluster nodes
- **Complete sample coverage**: Processes all samples without limitations
- **Sample ID preservation**: Maintains traceability throughout epochs
- **Layer-wise analysis**: Covers all Conv2d and Linear layers in ResNet18

### Dormant Unit Detection Algorithm

**Threshold-Based Detection**:
```python
def compute_per_sample_dormant(net, data_loader, device, dormant_unit_threshold=0.01):
    """
    Compute per-sample dormant proportions using epsilon threshold detection.
    
    Key insight: A neuron is dormant if its activation is effectively zero (≤ 1e-6)
    """
    eps = 1e-6  # Threshold for "effectively zero" activations
    
    # For Convolutional layers: [batch, channels, height, width]
    # A channel is dormant if ALL spatial locations are ≤ eps
    channel_max_per_sample = features.amax(dim=(2, 3))  # Max across H,W
    dormant_mask = (channel_max_per_sample <= eps)
    dormant_per_sample = dormant_mask.sum(dim=1).float()
    
    # For Fully Connected layers: [batch, features]
    # A neuron is dormant if its activation is ≤ eps
    dormant_mask = (features <= eps)
    dormant_per_sample = dormant_mask.sum(dim=1).float()
    
    # Convert counts to proportions
    per_sample_dormant_props = dormant_counts / total_neurons_per_sample
```

**Architecture Coverage**:
```python
# Hook registration for ResNet18
for name, module in net.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        hook = module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
        layer_idx += 1
```

### Usage and Integration

**Cluster Execution**:
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 multi_gpu_dormant_analysis.py \
    --start_epoch 0 --end_epoch 4000 --step 200

# Multi-node SLURM cluster
srun python multi_gpu_dormant_analysis.py \
    --start_epoch 0 --end_epoch 4000 --step 200
```

**Output Format**:
```python
# Per-epoch dormant proportion files
dormant_props_epoch_0.npy    # Shape: [num_samples]
dormant_props_epoch_200.npy  # Shape: [num_samples]
sample_ids_epoch_0.npy       # Shape: [num_samples] - for ID mapping
labels_epoch_0.npy           # Shape: [num_samples] - true labels
```

### Integration with Enhanced Sample Tracking

**Combined Analysis Pipeline**:
1. **Train models** with incremental learning + sample tracking
2. **Extract per-sample losses** using CSL pipeline
3. **Compute curvature metrics** for same samples
4. **Calculate dormant proportions** using multi-GPU analysis
5. **Correlate all metrics** using sample IDs:
   - CSL values per sample
   - Curvature metrics per sample
   - Memorization scores per sample
   - Dormant unit proportions per sample
   - Training dynamics (epoch/batch seen)

**Comprehensive Sample Analysis**:
```python
# Load all analysis results
csl_data = np.load("csl_cifar100_run1_noise_0.01.npy")
curvature_data = np.load("per_image_curvature.npy")
dormant_data = np.load("dormant_props_epoch_0.npy")
tracking_data = pd.read_csv("sample_tracking_data_final.csv")

# Merge all metrics by sample_id
comprehensive_analysis = tracking_data.merge(
    pd.DataFrame({
        'sample_id': range(len(csl_data)), 
        'csl': csl_data,
        'curvature': curvature_data,
        'dormant_prop': dormant_data
    }), on='sample_id'
)

# Multi-dimensional sample characterization
analysis_results = comprehensive_analysis.groupby(['class_id', 'task_num']).agg({
    'memo_score': ['mean', 'std'],
    'csl': ['mean', 'std'], 
    'curvature': ['mean', 'std'],
    'dormant_prop': ['mean', 'std']
})
```

**Advanced Sample Classification**:
- **High CSL + High Curvature + High Dormant + High Memo**: Most problematic samples
- **High CSL + Low Curvature + Low Dormant + High Memo**: Difficult but stable samples
- **Low CSL + High Curvature + High Dormant + Low Memo**: Easy but plasticity-impaired samples
- **Low CSL + Low Curvature + Low Dormant + Low Memo**: Ideal samples for training
- **Dormant Evolution**: Track how dormant proportions change across tasks
- **Plasticity Correlation**: Analyze dormant units vs memorization patterns

### Generalization for New Architectures

**Adaptation Requirements**:
1. **Update hook registration**: Modify for different layer types (e.g., BatchNorm, Attention)
2. **Adjust threshold values**: Fine-tune epsilon based on activation ranges
3. **Handle different shapes**: Adapt for Transformers, CNNs, or hybrid architectures
4. **Scale computation**: Adjust batch sizes and GPU memory usage

**Example for Vision Transformer**:
```python
# Modify hook registration for ViT
for name, module in net.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.MultiheadAttention)):
        hook = module.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
        layer_idx += 1

# Handle attention layers differently
if 'attention' in name.lower():
    # Attention-specific dormant detection
    attention_weights = features  # [batch, seq_len, embed_dim]
    dormant_mask = (attention_weights.abs().max(dim=1)[0] <= eps)
```

This dormant unit analysis provides crucial insights into neural network plasticity and can reveal which samples cause the most significant loss of representational capacity during continual learning.