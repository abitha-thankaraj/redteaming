#!/bin/bash

# Array of learning rates
# learning_rates=(1e-6 5e-7)
learning_rates=(1e-6)

# Array of value function experiments
# value_function_experiments=("multilabel" "binary")
value_function_experiments=("multilabel")

# Base master port
base_master_port=29500

# Path to your main SLURM script
main_script="/scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/rwr.sh"

# Loop through learning rates and value function experiments
for i in "${!learning_rates[@]}"; do
    for j in "${!value_function_experiments[@]}"; do
        lr=${learning_rates[$i]}
        value_fn=${value_function_experiments[$j]}
        master_port=$((base_master_port + i*10 + j))
        
        echo "Submitting job with learning rate: $lr, value function: $value_fn, and master port: $master_port"
        
        sbatch $main_script \
            "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            "defender" \
            "$master_port" \
            "weighted" \
            "$value_fn" \
            "overfit" \
            "$lr" \
            "1.0" \
            "Meta-Llama-3.1-8B-Instruct_length" \
            "overfit_value_fn_rwr_sweep_${value_fn}_${lr}_weighted"
    done
done