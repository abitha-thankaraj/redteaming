#!/bin/bash

#SBATCH --job-name=llama_train_multiturn_sft
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --partition=rl
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0


MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
AGENT_TYPE=("attacker" "defender")
MASTER_PORT=$((31003 + SLURM_ARRAY_TASK_ID))


bash /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/train_multiturn_sft.sh $MODEL_PATH ${AGENT_TYPE[$SLURM_ARRAY_TASK_ID]} $MASTER_PORT