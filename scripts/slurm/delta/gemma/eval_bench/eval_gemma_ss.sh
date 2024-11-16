#!/bin/bash
#SBATCH --job-name=eval_ss
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
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --time=03:30:00
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
EXPERIMENT_DESC=$4

ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135
ATTACKER_MODEL_TYPE="sft_trained_attacker"

# Simple safety tests evals
python $REPO_DIR/scripts/evaluate.py env=delta \
                dataset_configs=simplesafetytests \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC
