#!/bin/bash

# Test the new directory naming logic
test_dir_name() {
    local csv_path="$1"
    local name_desc="$2"
    
    # Extract CSV filename for custom CSV paths (without .csv extension)
    if [ -n "$csv_path" ]; then
        CSV_BASENAME=$(basename "$csv_path" .csv)
        # Remove 'sample_class_map_' prefix if present for cleaner names
        CSV_BASENAME=${CSV_BASENAME#sample_class_map_}
        # Remove 'distributed_' prefix if present for cleaner names  
        CSV_BASENAME=${CSV_BASENAME#distributed_}
        CSV_SUFFIX="_${CSV_BASENAME}"
    else
        CSV_SUFFIX=""
    fi
    
    # Standard parameters
    PREDETERMINED_SAMPLE_ORDER="ascending"
    CLASS_ORDER="random"
    SCRATCH_MODE="true"
    SINGLE_TASK_MODE="true"
    SINGLE_TASK_NUMBER=19
    SINGLE_TASK_EPOCHS=200
    EPOCHS_PER_TASK="200"
    INCREMENTAL_EPOCHS="false"
    
    DIR_NAME="results_predetermined_sample_${PREDETERMINED_SAMPLE_ORDER}_${CLASS_ORDER}${CSV_SUFFIX}"
    
    # Add scratch mode to directory name
    if [ "$SCRATCH_MODE" = "true" ]; then
        DIR_NAME="${DIR_NAME}_scratch"
    fi
    
    # Add epochs mode to directory name
    if [ "$INCREMENTAL_EPOCHS" = "true" ]; then
        DIR_NAME="${DIR_NAME}_incremental_${EPOCHS_PER_TASK}"
    else
        DIR_NAME="${DIR_NAME}_fixed_${EPOCHS_PER_TASK}"
    fi
    
    # Add single task mode identifier
    if [ "$SINGLE_TASK_MODE" = "true" ]; then
        DIR_NAME="${DIR_NAME}_single_task_${SINGLE_TASK_NUMBER}_${SINGLE_TASK_EPOCHS}epochs"
    fi
    
    echo "$name_desc:"
    echo "  $DIR_NAME"
    echo
}

echo "Testing new directory naming with different CSV files:"
echo "======================================================"

test_dir_name "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_distributed_stratified_interleaving_10strata.csv" "Stratified Interleaving"

test_dir_name "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_distributed_snake_25.csv" "Snake Pattern"

test_dir_name "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_distributed_uniform_50.csv" "Uniform Spacing"

test_dir_name "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_ascending.csv" "Default Ascending"

test_dir_name "" "No Custom CSV (Default)"