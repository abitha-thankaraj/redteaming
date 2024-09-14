#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam
module load cuda-12.3

# Arrays for the different configurations
defender_model_dirs=(
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-14-05-04-334/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-14-39-11-005/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-15-15-28-107/checkpoint-12/"
    "/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-10-15-49-31-596/checkpoint-12/"
)

defender_model_types=(
    "trained_defender_natural_lang_binary"
    "trained_defender_binary"
    "trained_defender_natural_lang_multiclass"
    "trained_defender_multilabel"
)

value_function_types=(
    "natural_lang_binary"
    "binary"
    "natural_lang_multiclass"
    "multilabel"
)

# Check if all arrays have the same length
if [ ${#defender_model_dirs[@]} -ne ${#defender_model_types[@]} ] || [ ${#defender_model_dirs[@]} -ne ${#value_function_types[@]} ]; then
    echo "Error: All arrays must have the same length"
    exit 1
fi

# Loop through the arrays
for i in "${!defender_model_dirs[@]}"; do
    echo "Running configuration $((i+1)) of ${#defender_model_dirs[@]}"
    
    # Run the Python script with the current configuration
    python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/evaluate.py \
        defender.model_dir="${defender_model_dirs[i]}" \
        defender.model_type="${defender_model_types[i]}" \
        value_function_evaluate.defender.value_function_type="${value_function_types[i]}"
    
    # Check if the Python script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Python script failed for configuration $((i+1))"
        exit 1
    fi
    
    echo "Configuration $((i+1)) completed successfully"
    echo "-------------------------------------------"
done

echo "All configurations completed successfully"