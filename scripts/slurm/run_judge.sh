#!/bin/bash
#SBATCH --job-name=judge_$SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/athankar/slurm/%A_%a.out
#SBATCH --error=/home/athankar/slurm/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=general
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-79%12

# Define directories
DIR1="/data/group_data/rl/datasets/redteaming/gen_eval_multiturn_attacks/"

# Define the pattern to search for within the file names
PATTERN="*.json"

# Find all files that match the pattern in the directory and store them in a variable
FILES=$(find $DIR1 -type f -name "$PATTERN")

# Filter out all files with 'advbench' in the filename
FILTERED_FILES=($(echo "$FILES" | grep -v "advbench"))
REPO_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming";

source ~/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

# Pass the correct file name to the Python script
python $REPO_DIR/redteam/data_generation/judge_multiturn_conversation.py multiturn_conversations_fname="${FILTERED_FILES[$SLURM_ARRAY_TASK_ID]}"
