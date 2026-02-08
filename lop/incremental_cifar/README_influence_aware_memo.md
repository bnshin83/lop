# Influence-Aware Memorization Scoring

This repository contains scripts to calculate influence-aware memorization scores that go beyond simple memorization to identify truly representative and impactful training samples.

## Problem Statement

Standard memorization scores alone can't identify samples that are critically important for a class because:

- **Mislabeled samples** may have high memorization but negative influence on test performance
- **Duplicate or near-duplicate samples** may be well-memorized but not representative of the class
- **Outlier samples** may be memorized but don't generalize well to test samples
- **Edge cases** may have high memorization but inconsistent influence across test samples

## Solution: Influence-Aware Memorization Scores

We propose four methods that combine memorization scores with influence function data to identify samples that are both well-memorized AND positively impactful on test performance.

## Methods

### 1. Influence-Weighted Memorization
**Formula:** `memorization_score × mean_positive_influence`

**What it identifies:** Samples that are both well-memorized AND have strong positive influence on test samples.

**Use case:** Finding samples that are learned well and help the model perform better on test data.

```python
# Calculate positive influence (set negative values to 0)
pos_infl_mean = np.maximum(influence_matrix, 0).mean(axis=1)
influence_weighted = memorization_score * pos_infl_mean
```

### 2. Representativeness Score
**Formula:** `memorization_score × (positive_influence / influence_std)`

**What it identifies:** Samples with consistent positive influence across test samples.

**Use case:** Finding representative samples that consistently help the model, avoiding outliers that might strongly influence only specific test cases.

```python
# Calculate consistency (lower std = more consistent)
influence_std = influence_matrix.std(axis=1)
representativeness = memorization_score * (pos_infl_mean / influence_std)
```

### 3. Weighted Linear Combination
**Formula:** `α × memorization_score + β × normalized_influence` (default: α=0.7, β=0.3)

**What it identifies:** Balanced importance considering both memorization and influence factors.

**Use case:** When you want controllable mixing of memorization and influence importance.

```python
# Normalize influence to [0,1] range
norm_influence = pos_infl_mean / pos_infl_mean.max()
weighted_combination = 0.7 * memorization_score + 0.3 * norm_influence
```

### 4. Critical Sample Detection
**Formula:** `memorization_score × (1 - influence_rank_normalized)`

**What it identifies:** Samples that rank highest in influence within their class and are well-memorized.

**Use case:** Identifying the most critical samples for each class based on their influence ranking.

```python
# Rank by influence (higher influence = lower rank number)
influence_ranks = np.argsort(np.argsort(-pos_infl_mean))
rank_normalized = influence_ranks / (len(influence_ranks) - 1)
critical_score = memorization_score * (1 - rank_normalized)
```

## Key Concepts

### Influence Standard Deviation (`influence_std`)
- **Low `influence_std`:** Sample has **consistent** influence across test samples → Good representative
- **High `influence_std`:** Sample has **variable** influence → Potentially an outlier or edge case

### Positive Influence Filtering
All methods use `np.maximum(influence_matrix, 0)` to focus only on positive influences, filtering out samples that hurt test performance.

## Files

### Scripts

1. **`calculate_influence_aware_memo.py`** - Main script to calculate all four influence-aware scores
2. **`update_memo_stats_csv.py`** - Updates existing memorization stats CSV with new scores

### Data Files

1. **`cifar100/cifar100_infl_matrix.npz`** - Input influence matrix data containing:
   - `tr_mem`: Training memorization scores (50,000 samples)
   - `infl_matrix_classX`: Influence matrices for each class (500×100 each)
   - `tr_classidx_X`: Training sample indices for each class

2. **`cifar100/cifar100_influence_aware_memo.npz`** - Output containing all calculated scores

3. **`cifar100/per_class_memorization_stats.csv`** - Updated CSV with statistical summaries

## Usage

### Calculate Influence-Aware Scores

```bash
conda activate lop
python calculate_influence_aware_memo.py \
    --input cifar100/cifar100_infl_matrix.npz \
    --output cifar100/cifar100_influence_aware_memo.npz \
    --analyze-class 0
```

**Parameters:**
- `--input`: Path to influence matrix .npz file
- `--output`: Path to save calculated scores (optional)
- `--analyze-class`: Class to analyze in detail (default: 0)
- `--top-k`: Number of top samples to show in analysis (default: 10)

### Update CSV with New Scores

```bash
python update_memo_stats_csv.py \
    --influence-file cifar100/cifar100_influence_aware_memo.npz \
    --csv-file cifar100/per_class_memorization_stats.csv \
    --output cifar100/updated_stats.csv
```

**Parameters:**
- `--influence-file`: Path to calculated influence-aware scores
- `--csv-file`: Path to existing memorization stats CSV
- `--output`: Path for updated CSV (optional, defaults to overwriting input)

## Output Format

### NPZ File Contents
```python
{
    'influence_weighted_memo': {class_id: scores_array},
    'representativeness_score': {class_id: scores_array}, 
    'weighted_combination': {class_id: scores_array},
    'critical_sample_score': {class_id: scores_array},
    'training_indices': {class_id: indices_array},
    'original_memo': {class_id: memo_scores_array}
}
```

### CSV File Columns

**Original columns:** `class_label`, `sample_count`, `mean_memo_score`, `std_memo_score`, `min_memo_score`, `max_memo_score`, `median_memo_score`

**New columns for each method:**
- **Influence-Weighted:** `mean_infl_weighted_memo`, `std_infl_weighted_memo`, `min_infl_weighted_memo`, `max_infl_weighted_memo`, `median_infl_weighted_memo`
- **Representativeness:** `mean_representativeness`, `std_representativeness`, `min_representativeness`, `max_representativeness`, `median_representativeness`  
- **Weighted Combination:** `mean_weighted_combination`, `std_weighted_combination`, `min_weighted_combination`, `max_weighted_combination`, `median_weighted_combination`
- **Critical Sample:** `mean_critical_sample`, `std_critical_sample`, `min_critical_sample`, `max_critical_sample`, `median_critical_sample`

## Example Results

### Score Ranges (CIFAR-100)
- **Original memorization:** 0.0829 - 0.5842
- **Influence-weighted:** 0.000151 - 0.003366 (smaller scale due to multiplication)
- **Representativeness:** 0.0175 - 0.2034  
- **Weighted combination:** 0.1103 - 0.5846 (similar to original, rebalanced)
- **Critical sample:** 0.000000 - 1.000000

### Sample Analysis (Class 0)
```
Top samples by Original Memorization:
  1. Sample 23137: memo=1.0000, infl_w=0.004835, repr=0.232562, crit=0.995992
  
Top samples by Influence-Weighted Memorization:  
  1. Sample 33950: memo=0.9030, infl_w=0.007867, repr=0.154885, crit=0.902954
  
Top samples by Representativeness Score:
  1. Sample  1828: memo=1.0000, infl_w=0.004219, repr=0.344178, crit=0.985972
```

## Key Insights

1. **Rankings differ significantly** across methods, showing each captures different aspects of sample importance
2. **High memorization ≠ high influence** - Sample 33950 has lower memorization (0.903) but highest influence-weighted score
3. **Consistency matters** - Sample 1828 leads representativeness due to consistent positive influence
4. **Top classes remain consistent** - Classes 72, 55, 35, 11, 44 consistently rank high across all methods

## Applications

### Data Pruning
Use influence-aware scores to identify which samples to keep when reducing dataset size.

### Active Learning
Prioritize samples for labeling based on their expected impact on model performance.

### Data Quality Assessment
Identify potentially mislabeled or low-quality samples with high memorization but low/negative influence.

### Curriculum Learning
Order training samples from most to least representative within each class.

## Dependencies

- numpy
- pandas (for CSV operations)
- argparse (for command-line interface)

## Notes

- All influence matrices are processed per-class (500 training samples → 100 test samples per class)
- Negative influences are filtered out using `np.maximum(influence_matrix, 0)`
- Statistical measures (mean, std, min, max, median) are calculated for each class
- The scripts handle edge cases like division by zero and missing data gracefully