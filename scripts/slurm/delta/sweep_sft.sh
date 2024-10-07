#!/bin/bash

# Array of learning rates
learning_rates=(1e-6 5e-7 1e-7)
# Base master port
base_master_port=29520

# Path to your main SLURM script
main_script="/scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/sft_iter_1.sh"

# Loop through learning rates
for i in "${!learning_rates[@]}"; do
    lr=${learning_rates[$i]}
    master_port=$((base_master_port + i))
    
    echo "Submitting job with learning rate: $lr and master port: $master_port"
    
    sbatch $main_script \
        "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        "defender" \
        "$master_port" \
        "naive_balance" \
        "" \
        "" \
        "$lr" \
        "1.0" \
        "Meta-Llama-3.1-8B-Instruct_length" \
        "iter_1_sft_sweep_${lr}" \
        "6198"
done


# #!/bin/bash

# # Array of learning rates
# learning_rates=(1e-6)
# # Base master port
# base_master_port=29520

# # Path to your main SLURM script
# main_script="/scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/sft_attacker_iter_1.sh"

# # Loop through learning rates
# for i in "${!learning_rates[@]}"; do
#     lr=${learning_rates[$i]}
#     master_port=$((base_master_port + i))
    
#     echo "Submitting job with learning rate: $lr and master port: $master_port"
    
#     sbatch $main_script \
#         "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#         "defender" \
#         "$master_port" \
#         "naive_balance" \
#         "" \
#         "" \
#         "$lr" \
#         "1.0" \
#         "Meta-Llama-3.1-8B-Instruct_length" \
#         "attacker_iter_1_sft_sweep_${lr}" \
#         "6198"
# done