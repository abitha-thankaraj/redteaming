#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=bcgv-delta-gpu
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --gpus-per-node=3
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --time=04:00:00

source ~/.bashrc
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
module load cuda/12.3.0
conda activate redteam

# Loads common environment variables
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

TEMPERATURE=$1

# Get the output path from the first job for defender
OUTPUT_PATH=$2

echo "Using output from previous job: $OUTPUT_PATH"

# Find the latest checkpoint directory inside the output path
LATEST_CHECKPOINT=$(ls -td $OUTPUT_PATH/checkpoint-* | head -1)

# Check if a checkpoint was found
if [ -z "$LATEST_CHECKPOINT" ]; then
  echo "No checkpoint found in $OUTPUT_PATH"
  exit 1
fi

echo "Latest checkpoint directory: $LATEST_CHECKPOINT"

DEFENDER_MODEL_TYPE=$3
ATTACKER_MODEL_DIR=$4
ATTACKER_MODEL_TYPE=$5
EXPERIMENT_DESC=$6
# Openai evals
python $REPO_DIR/scripts/evaluate.py env=delta \
                dataset_configs=openai \
                dataset_configs.dataset_path=$DATA_DIR/cfg_multiturn_generation_prompts/json_files/gpt4_redteaming_questions.json \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC
# Jailbreakbench evals
python $REPO_DIR/scripts/evaluate.py env=delta \
                dataset_configs=jailbreakbench \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC
