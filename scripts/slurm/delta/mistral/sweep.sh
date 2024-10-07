#!/bin/bash

# Array of learning rates
learning_rates=(5e-7 1e-7)

# Base master port
base_master_port=29500

# Path to your main SLURM script
main_script="/scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/mistral/rwr.sh"

# Loop through learning rates
for i in "${!learning_rates[@]}"; do
    lr=${learning_rates[$i]}
    master_port=$((base_master_port + i))
    
    echo "Submitting job with learning rate: $lr and master port: $master_port"
    
    sbatch $main_script \
        "mistralai/Mistral-7B-Instruct-v0.1" \
        "defender" \
        "$master_port" \
        "naive_balance" \
        "" \
        "" \
        "$lr" \
        "1.0" \
        "Mistral-7B-Instruct-v0.1_length" \
        "mistral_rwr_sweep_naive_balance_${lr}"
done