#!/bin/bash
#SBATCH --job-name=eval_jbb
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.err
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24 
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --partition=preempt
#SBATCH --time=02:30:00


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env
TEMPERATURE=$1

# Get the output path from the first job for defender
LATEST_CHECKPOINT=$2
DEFENDER_MODEL_TYPE=$3
EXPERIMENT_DESC=$4

ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135
ATTACKER_MODEL_TYPE="sft_trained_attacker"


# Jailbreakbench evals
python $REPO_DIR/scripts/evaluate.py env=babel \
                dataset_configs=jailbreakbench \
                attacker.model_dir=$ATTACKER_MODEL_DIR \
                attacker.model_type=$ATTACKER_MODEL_TYPE \
                defender.model_dir=$LATEST_CHECKPOINT \
                defender.model_type=$DEFENDER_MODEL_TYPE \
                defender.generation_kwargs.temperature=$TEMPERATURE \
                experiment_desc=$EXPERIMENT_DESC
