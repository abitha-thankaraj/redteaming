#!/bin/bash

# Array of VALUE_FUNCTION_TYPE values
value_function_types=("natural_lang_binary" "binary" "natural_lang_multiclass" "multilabel")

# Array of VALUE_FUNCTION_EXPERIMENT values
value_function_experiments=("prefix" "overfit")

# Path to the original script
original_script="/data/user_data/athankar/redteaming/scripts/train_multiturn_rwr.sh"

# Loop through all combinations
for vft in "${value_function_types[@]}"; do
    for vfe in "${value_function_experiments[@]}"; do
        echo "Running job with VALUE_FUNCTION_TYPE=$vft and VALUE_FUNCTION_EXPERIMENT=$vfe"
        
        # Execute the original script with the current combination of parameters
        bash "$original_script" "$vft" "$vfe"
        
        # Optional: add a delay between job launches if needed
        # sleep 10
    done
done

echo "All jobs have been launched."