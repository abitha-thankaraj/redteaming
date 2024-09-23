#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.err
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
#SBATCH --exclude=gpub054


source ~/.bashrc
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
module load cuda/12.3.0
conda activate redteam

# Loads common environment variables
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

TEMPERATURE=$1

# Get the output path from the first job for defender
LATEST_CHECKPOINT=$2

DEFENDER_MODEL_TYPE=$3
ATTACKER_MODEL_DIR=$4
ATTACKER_MODEL_TYPE=$5
EXPERIMENT_DESC=$6
VALUE_FUNCTION_TYPE=$7

echo "TEMPERATURE: $TEMPERATURE"
echo "LATEST_CHECKPOINT: $LATEST_CHECKPOINT"
echo "DEFENDER_MODEL_TYPE: $DEFENDER_MODEL_TYPE"
echo "ATTACKER_MODEL_DIR: $ATTACKER_MODEL_DIR"
echo "ATTACKER_MODEL_TYPE: $ATTACKER_MODEL_TYPE"
echo "EXPERIMENT_DESC: $EXPERIMENT_DESC"
echo "VALUE_FUNCTION_TYPE: $VALUE_FUNCTION_TYPE"



# Openai evals
python $REPO_DIR/scripts/evaluate.py env=delta \
                dataset_configs=openai \
                dataset_configs.dataset_path=$DATA_DIR/cfg_multiturn_generation_prompts/json_files/gpt4_redteaming_questions.json \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC \
                value_function_evaluate.do_eval=True \
                value_function_evaluate.defender.evaluate=True \
                value_function_evaluate.defender.value_function_type=$VALUE_FUNCTION_TYPE



# Jailbreakbench evals
python $REPO_DIR/scripts/evaluate.py env=delta \
                dataset_configs=jailbreakbench \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC \
                value_function_evaluate.do_eval=True \
                value_function_evaluate.defender.evaluate=True \
                value_function_evaluate.defender.value_function_type=$VALUE_FUNCTION_TYPE

