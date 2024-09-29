#!/bin/bash

source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

LATEST_CHECKPOINTS=(
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-23-13-38-37-446/checkpoint-80"
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-23-13-38-37-017/checkpoint-80"
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-23-13-38-37-972/checkpoint-80/"
    "/scratch/bcgv/models/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-23-13-38-37-931/checkpoint-80"
)
LEARNING_RATES=(5e-6 1e-6 5e-7 1e-7)


SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135

RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker"
RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183
RWR_DEFENDER_MODEL_NAME="raw_rewards_rwr_trained_defender"



for i in "${!LATEST_CHECKPOINTS[@]}"; do
    checkpoint="${LATEST_CHECKPOINTS[$i]}"
    lr="${LEARNING_RATES[$i]}"

    for temperature in 0.0 0.7 1.0; do
        EXPERIMENT_DESC="rwr_raw_rewards_${lr}_${temperature}"
        sbatch $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $checkpoint $RWR_DEFENDER_MODEL_NAME $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC

        # sbatch $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $checkpoint $RWR_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC
    done
done