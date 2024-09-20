#!/bin/bash


source ~/.bashrc
# source /sw/external/python/anaconda3/etc/profile.d/conda.sh
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env
# Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature
EXPERIMENT_DESC="untrained_defender"
UNTRAINED_DEFENDER_MODEL_NAME="untrained_defender"
# Model parent dir is the parent directory for the checkpoint folder 
# The checkpoint folder is where the model is loaded from
UNTRAINED_DEFENDER_MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.1"

SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker_mistral"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135
RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker_mistral"
RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366



# Args: $DEFENDER_MODEL_PARENT_DIR $TEMPERATURE $DEFENDER_MODEL_NAME $ATTACKER_MODEL_DIR $ATTACKER_MODEL_NAME
# for loop through temperatures
for temperature in 0.0 0.7 1.0
do
    sbatch $REPO_DIR/scripts/slurm/delta/mistral/evaluate_iter_0.sh $temperature $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE "$EXPERIMENT_DESC.$SFT_ATTACKER_MODEL_TYPE"
    sbatch $REPO_DIR/scripts/slurm/delta/mistral/evaluate_iter_0.sh $temperature $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE "$EXPERIMENT_DESC.$RWR_ATTACKER_MODEL_TYPE"
done
