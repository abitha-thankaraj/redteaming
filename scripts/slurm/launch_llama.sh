#!/bin/bash

#SBATCH --job-name=llama_$SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/ftajwar/slurm/%A_%a.out
#SBATCH --error=/home/ftajwar/slurm/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8

# Define directories
HARMBENCH_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/harmbench_chunked/"
OPENAI_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/openai_chunked/"
ADVBENCH_DIR="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/advbench_chunked"

# Find all files that match the pattern in both directories and store them as an array
FILES=($(find $HARMBENCH_DIR $OPENAI_DIR $ADVBENCH_DIR-type f -name "*.json"))
# FILES=($(find $ADVBENCH_DIR -type f -name "*.json"))


# Variables
FAST_CHAT_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct";
FAST_CHAT_API_PORT=$((11003 + SLURM_ARRAY_TASK_ID))
CONTROLLER_PORT=$((12000 + SLURM_ARRAY_TASK_ID))
WORKER_PORT=$((13000 + SLURM_ARRAY_TASK_ID))

REPO_DIR="/home/ftajwar/redteaming"; #TODO: FAHIM
CHAT_COMPLETION_MODEL="llama-3.1-8b-instruct.yaml";

# Navigate to the script directory
cd $REPO_DIR/scripts/slurm/

# Deploy and infer
# ./deploy_and_infer.sh $FAST_CHAT_MODEL_PATH $FAST_CHAT_API_PORT $REPO_DIR $CHAT_COMPLETION_MODEL "${FILES[$SLURM_ARRAY_TASK_ID]}" $CONTROLLER_PORT $WORKER_PORT
./deploy_and_infer_ft.sh $FAST_CHAT_MODEL_PATH $FAST_CHAT_API_PORT $REPO_DIR $CHAT_COMPLETION_MODEL "${MISSING_FNAMES[$SLURM_ARRAY_TASK_ID]}" $CONTROLLER_PORT $WORKER_PORT
