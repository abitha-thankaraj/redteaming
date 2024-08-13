#!/bin/bash
#SBATCH --job-name=judge_$SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/ftajwar/slurm/%A_%a.out
#SBATCH --error=/home/ftajwar/slurm/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-169%24

# Define directories
DIR1="/data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks_no_special_tokens/"

# Define the pattern to search for within the file names
PATTERN="*.json"

# # Find all files that match the pattern in the directory and store them in a variable
FILES=($(find $DIR1 -type f -name "$PATTERN"))
# DATASET_NAME="advbench"

# # # Filter out all files with 'advbench' in the filename
# FILTERED_FILES=($(echo "$FILES" | grep -v $DATASET_NAME))
REPO_DIR="/home/ftajwar/redteaming";

source /home/ftajwar/.bashrc
source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate fastchat

# Pass the correct file name to the Python script
python $REPO_DIR/scripts/data_generation/judge_multiturn_conversation.py multiturn_conversations_fname="${FILES[$SLURM_ARRAY_TASK_ID]}" 
# prompt_dataset=$DATASET_NAME