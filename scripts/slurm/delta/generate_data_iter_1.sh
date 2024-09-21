#!/bin/bash


source ~/.bashrc
# source /sw/external/python/anaconda3/etc/profile.d/conda.sh
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env
# Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature
EXPERIMENT_DESC="iter_1"

UNTRAINED_DEFENDER_MODEL_TYPE="untrained_defender"
UNTRAINED_DEFENDER_MODEL_DIR="meta-llama/Meta-Llama-3.1-8B-Instruct"

SFT_DEFENDER_MODEL_TYPE="sft_defender"
SFT_DEFENDER_MODEL_DIR="$MODEL_PARENT_DIR/multiturn_sft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-17-09-40-28-211/checkpoint-130"

SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker"
SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135


RWR_DEFENDER_MODEL_TYPE="rwr_defender"
RWR_DEFENDER_MODEL_DIR="$MODEL_PARENT_DIR/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-09-17-09-40-12-223/checkpoint-80"

RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker"
RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183


for chunk in {0..30}
do
    sbatch $REPO_DIR/scripts/slurm/delta/generate_data_iter_0.sh 0.7 $UNTRAINED_DEFENDER_MODEL_DIR $UNTRAINED_DEFENDER_MODEL_TYPE $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE "$RWR_ATTACKER_MODEL_TYPE.$UNTRAINED_DEFENDER_MODEL_TYPE.iter0.$chunk" $chunk
    sbatch $REPO_DIR/scripts/slurm/delta/generate_data_iter_0.sh 0.7 $SFT_DEFENDER_MODEL_DIR $SFT_DEFENDER_MODEL_TYPE $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE "$SFT_ATTACKER_MODEL_TYPE.$SFT_DEFENDER_MODEL_TYPE.iter1.$chunk" $chunk
    sbatch $REPO_DIR/scripts/slurm/delta/generate_data_iter_0.sh 0.7 $RWR_DEFENDER_MODEL_DIR $RWR_DEFENDER_MODEL_TYPE $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE "$RWR_ATTACKER_MODEL_TYPE.$RWR_DEFENDER_MODEL_TYPE.iter1.$chunk" $chunk
done
