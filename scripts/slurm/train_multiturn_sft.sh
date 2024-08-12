#!/bin/bash

#SBATCH --job-name=mistral_train_multiturn_sft
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:A6000:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --partition=general
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0


MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# AGENT_TYPE=("attacker" "defender")
AGENT_TYPE=("defender")

MASTER_PORT=$((31003 + SLURM_ARRAY_TASK_ID))


bash /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/train_multiturn_sft.sh $MODEL_PATH "${AGENT_TYPE[$SLURM_ARRAY_TASK_ID]}" $MASTER_PORT