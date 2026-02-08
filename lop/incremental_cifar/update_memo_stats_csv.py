import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def calculate_stats(scores):
    """Calculate statistical measures for a set of scores"""
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores)
    }

def update_memorization_stats_csv(influence_aware_file, csv_file, output_csv=None):
    """
    Update the per-class memorization stats CSV with influence-aware scores
    
    Args:
        influence_aware_file: Path to the influence-aware memo scores .npz file
        csv_file: Path to existing CSV file
        output_csv: Path for updated CSV (optional, defaults to overwriting input)
    """
    
    # Load influence-aware scores
    print(f"Loading influence-aware scores from {influence_aware_file}")
    data = np.load(influence_aware_file, allow_pickle=True)
    
    # Load existing CSV
    print(f"Loading existing CSV from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Initialize new columns
    new_columns = [
        'mean_infl_weighted_memo', 'std_infl_weighted_memo', 'min_infl_weighted_memo', 
        'max_infl_weighted_memo', 'median_infl_weighted_memo',
        
        'mean_representativeness', 'std_representativeness', 'min_representativeness',
        'max_representativeness', 'median_representativeness',
        
        'mean_weighted_combination', 'std_weighted_combination', 'min_weighted_combination',
        'max_weighted_combination', 'median_weighted_combination',
        
        'mean_critical_sample', 'std_critical_sample', 'min_critical_sample',
        'max_critical_sample', 'median_critical_sample'
    ]
    
    # Initialize new columns with NaN
    for col in new_columns:
        df[col] = np.nan
    
    print("Calculating statistics for all classes...")
    
    # Process each class
    for class_id in range(100):
        if class_id % 20 == 0:
            print(f"Processing class {class_id}/100")
            
        # Get scores for this class
        infl_weighted = data['influence_weighted_memo'].item()[class_id]
        representativeness = data['representativeness_score'].item()[class_id]
        weighted_combination = data['weighted_combination'].item()[class_id]
        critical_sample = data['critical_sample_score'].item()[class_id]
        
        # Calculate statistics
        infl_weighted_stats = calculate_stats(infl_weighted)
        represent_stats = calculate_stats(representativeness)
        weighted_comb_stats = calculate_stats(weighted_combination)
        critical_sample_stats = calculate_stats(critical_sample)
        
        # Find the row for this class
        class_row_idx = df[df['class_label'] == class_id].index[0]
        
        # Update influence-weighted memorization columns
        df.loc[class_row_idx, 'mean_infl_weighted_memo'] = infl_weighted_stats['mean']
        df.loc[class_row_idx, 'std_infl_weighted_memo'] = infl_weighted_stats['std']
        df.loc[class_row_idx, 'min_infl_weighted_memo'] = infl_weighted_stats['min']
        df.loc[class_row_idx, 'max_infl_weighted_memo'] = infl_weighted_stats['max']
        df.loc[class_row_idx, 'median_infl_weighted_memo'] = infl_weighted_stats['median']
        
        # Update representativeness columns
        df.loc[class_row_idx, 'mean_representativeness'] = represent_stats['mean']
        df.loc[class_row_idx, 'std_representativeness'] = represent_stats['std']
        df.loc[class_row_idx, 'min_representativeness'] = represent_stats['min']
        df.loc[class_row_idx, 'max_representativeness'] = represent_stats['max']
        df.loc[class_row_idx, 'median_representativeness'] = represent_stats['median']
        
        # Update weighted combination columns
        df.loc[class_row_idx, 'mean_weighted_combination'] = weighted_comb_stats['mean']
        df.loc[class_row_idx, 'std_weighted_combination'] = weighted_comb_stats['std']
        df.loc[class_row_idx, 'min_weighted_combination'] = weighted_comb_stats['min']
        df.loc[class_row_idx, 'max_weighted_combination'] = weighted_comb_stats['max']
        df.loc[class_row_idx, 'median_weighted_combination'] = weighted_comb_stats['median']
        
        # Update critical sample columns
        df.loc[class_row_idx, 'mean_critical_sample'] = critical_sample_stats['mean']
        df.loc[class_row_idx, 'std_critical_sample'] = critical_sample_stats['std']
        df.loc[class_row_idx, 'min_critical_sample'] = critical_sample_stats['min']
        df.loc[class_row_idx, 'max_critical_sample'] = critical_sample_stats['max']
        df.loc[class_row_idx, 'median_critical_sample'] = critical_sample_stats['median']
    
    # Determine output path
    if output_csv is None:
        output_csv = csv_file
    
    # Save updated CSV
    print(f"Saving updated CSV to {output_csv}")
    df.to_csv(output_csv, index=False, float_format='%.6f')
    
    print("CSV update complete!")
    
    # Print summary comparison
    print("\n=== Summary Comparison ===")
    print("Original Memorization vs Influence-Aware Scores:")
    print(f"Original memo mean range: {df['mean_memo_score'].min():.4f} - {df['mean_memo_score'].max():.4f}")
    print(f"Infl-weighted memo mean range: {df['mean_infl_weighted_memo'].min():.6f} - {df['mean_infl_weighted_memo'].max():.6f}")
    print(f"Representativeness mean range: {df['mean_representativeness'].min():.4f} - {df['mean_representativeness'].max():.4f}")
    print(f"Weighted combination mean range: {df['mean_weighted_combination'].min():.4f} - {df['mean_weighted_combination'].max():.4f}")
    
    # Show classes with highest scores
    print(f"\nTop 5 classes by original memorization:")
    top_orig = df.nlargest(5, 'mean_memo_score')[['class_label', 'mean_memo_score']]
    for _, row in top_orig.iterrows():
        print(f"  Class {int(row['class_label'])}: {row['mean_memo_score']:.4f}")
    
    print(f"\nTop 5 classes by influence-weighted memorization:")
    top_infl = df.nlargest(5, 'mean_infl_weighted_memo')[['class_label', 'mean_infl_weighted_memo']]
    for _, row in top_infl.iterrows():
        print(f"  Class {int(row['class_label'])}: {row['mean_infl_weighted_memo']:.6f}")
    
    print(f"\nTop 5 classes by representativeness:")
    top_repr = df.nlargest(5, 'mean_representativeness')[['class_label', 'mean_representativeness']]
    for _, row in top_repr.iterrows():
        print(f"  Class {int(row['class_label'])}: {row['mean_representativeness']:.4f}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update memorization stats CSV with influence-aware scores')
    parser.add_argument('--influence-file', '-i', required=True, 
                       help='Path to influence-aware memo scores .npz file')
    parser.add_argument('--csv-file', '-c', required=True,
                       help='Path to existing per-class memorization stats CSV')
    parser.add_argument('--output', '-o', 
                       help='Path for updated CSV (optional, defaults to overwriting input)')
    
    args = parser.parse_args()
    
    # Update the CSV
    update_memorization_stats_csv(args.influence_file, args.csv_file, args.output)