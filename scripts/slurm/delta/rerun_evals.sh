#!/bin/bash

source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env
# Define the arrays
LATEST_CHECKPOINTS=(
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-20-09-55-14-661/checkpoint-80"
    # "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-19-00-53-20-607/checkpoint-80"
    # "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-19-00-49-34-880/checkpoint-80"
    # "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-19-04-38-03-478/checkpoint-80"
    # "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-19-00-49-35-105/checkpoint-80"
)

EXPERIMENT_DESCS=(
    "overfit_value_fn_rwr_sweep_multilabel_1e-6_weighted"
    # "value_fn_rwr_sweep_multilabel_5e-7_weighted_rerun"
    # "value_fn_rwr_sweep_multilabel_1e-6_weighted_rerun"
    # "value_fn_rwr_sweep_binary_5e-7_weighted_rerun"
    # "value_fn_rwr_sweep_binary_1e-6_weighted_rerun"
)

# Set default values for other parameters
TEMPERATURE=1.
DEFENDER_MODEL_TYPE="overfit_1_value_fn_rwr_trained_attacker"
ATTACKER_MODEL_TYPE="sft_trained_attacker"
ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135

# ATTACKER_MODEL_DIR="/path/to/attacker/model"
# ATTACKER_MODEL_TYPE="attacker-model-type"

# Check if arrays have the same length
if [ ${#LATEST_CHECKPOINTS[@]} -ne ${#EXPERIMENT_DESCS[@]} ]; then
    echo "Error: The number of checkpoints does not match the number of experiment descriptions."
    exit 1
fi

# Loop through the arrays using indices
for i in "${!LATEST_CHECKPOINTS[@]}"; do
    checkpoint="${LATEST_CHECKPOINTS[$i]}"
    desc="${EXPERIMENT_DESCS[$i]}"

    echo "Running evaluation with:"
    echo "LATEST_CHECKPOINT: $checkpoint"
    echo "EXPERIMENT_DESC: $desc"
    
    # Call the evaluation script
    sbatch /scratch/bcgv/athankaraj/redteaming/scripts/slurm/delta/evaluate_iter_0.sh $TEMPERATURE "$checkpoint" $DEFENDER_MODEL_TYPE $ATTACKER_MODEL_DIR $ATTACKER_MODEL_TYPE "$desc"
    
    # Optional: add a small delay between job submissions
    sleep 1
done