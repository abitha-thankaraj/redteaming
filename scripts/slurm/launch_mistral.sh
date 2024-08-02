#!/bin/bash

#SBATCH --job-name=mistral_$SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/athankar/slurm/%A_%a.out
#SBATCH --error=/home/athankar/slurm/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-38%13

# Define directories
HARMBENCH_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/harmbench_chunked/"
OPENAI_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/openai_chunked/"

# Find all files that match the pattern in both directories and store them as an array
FILES=($(find $HARMBENCH_DIR $OPENAI_DIR -type f -name "*.json"))

# Variables
FAST_CHAT_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1";
FAST_CHAT_API_PORT=$((9003 + SLURM_ARRAY_TASK_ID))
CONTROLLER_PORT=$((23000 + SLURM_ARRAY_TASK_ID))
WORKER_PORT=$((24000 + SLURM_ARRAY_TASK_ID))

REPO_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming"; #TODO: FAHIM
CHAT_COMPLETION_MODEL="mistral-7b-instruct-v0.1.yaml";

# Navigate to the script directory
cd $REPO_DIR/scripts/slurm/

# Deploy and infer
./deploy_and_infer_ft.sh $FAST_CHAT_MODEL_PATH $FAST_CHAT_API_PORT $REPO_DIR $CHAT_COMPLETION_MODEL "${FILES[$SLURM_ARRAY_TASK_ID]}" $CONTROLLER_PORT $WORKER_PORT
