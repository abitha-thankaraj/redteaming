#!/bin/bash

source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

LATEST_CHECKPOINTS=(
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-22-21-25-33-097/checkpoint-80"
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-22-21-25-33-180/checkpoint-80"
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-22-21-33-06-451/checkpoint-80"
)
LEARNING_RATES=(5e-6 1e-6 5e-7)


SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135

RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker"
RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183
RWR_DEFENDER_MODEL_NAME="value_fn_rwr_trained_defender"

VALUE_FUNCTION_TYPE="natural_lang_multiclass"


for i in "${!LATEST_CHECKPOINTS[@]}"; do
    checkpoint="${LATEST_CHECKPOINTS[$i]}"

    for temperature in 0.0 0.7 1.0; do
        EXPERIMENT_DESC="prefix_value_fn_rwr_sweep_${VALUE_FUNCTION_TYPE}_${LEARNING_RATES[$i]}_weighted_raw_rewards_${temperature}"

        sbatch $REPO_DIR/scripts/slurm/delta/value_fn_exps/evaluate_value_fn_exps.sh $temperature $checkpoint $RWR_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC $VALUE_FUNCTION_TYPE
    done
done