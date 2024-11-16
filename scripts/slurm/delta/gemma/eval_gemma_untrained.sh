#!/bin/bash


source ~/.bashrc
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env
# Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature
EXPERIMENT_DESC=""
UNTRAINED_DEFENDER_MODEL_NAME="untrained_defender"
# Model parent dir is the parent directory for the checkpoint folder 
# The checkpoint folder is where the model is loaded from
UNTRAINED_DEFENDER_MODEL_DIR="google/gemma-2-2b-it"

# Args: $DEFENDER_MODEL_PARENT_DIR $TEMPERATURE $DEFENDER_MODEL_NAME $ATTACKER_MODEL_DIR $ATTACKER_MODEL_NAME
# for loop through temperatures
for temperature in 0.0 0.7 1.0
do
    sbatch $REPO_DIR/scripts/slurm/delta/gemma/eval_bench/eval_gemma_oai.sh $temperature $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.$SFT_ATTACKER_MODEL_TYPE.temp$temperature"
    sbatch $REPO_DIR/scripts/slurm/delta/gemma/eval_bench/eval_gemma_jbb.sh $temperature $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.$SFT_ATTACKER_MODEL_TYPE.temp$temperature"
    sbatch $REPO_DIR/scripts/slurm/delta/gemma/eval_bench/eval_gemma_ss.sh $temperature $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.$SFT_ATTACKER_MODEL_TYPE.temp$temperature"
done
