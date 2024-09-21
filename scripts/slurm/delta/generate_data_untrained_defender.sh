#!/bin/bash


source ~/.bashrc
# source /sw/external/python/anaconda3/etc/profile.d/conda.sh
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env
# Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature
EXPERIMENT_DESC="untrained_defender"
UNTRAINED_DEFENDER_MODEL_NAME="untrained_defender"
# Model parent dir is the parent directory for the checkpoint folder 
# The checkpoint folder is where the model is loaded from
UNTRAINED_DEFENDER_MODEL_DIR="meta-llama/Meta-Llama-3.1-8B-Instruct"

SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135


# Args: $DEFENDER_MODEL_PARENT_DIR $TEMPERATURE $DEFENDER_MODEL_NAME $ATTACKER_MODEL_DIR $ATTACKER_MODEL_NAME
# for loop through temperatures
# for chunk in 0... 30
for chunk in {0..30}
do
    sbatch $REPO_DIR/scripts/slurm/delta/generate_data_iter_0.sh 0.7 $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE "$EXPERIMENT_DESC.$SFT_ATTACKER_MODEL_TYPE.iter0.$chunk" $chunk
done
