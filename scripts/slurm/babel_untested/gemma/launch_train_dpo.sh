#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.err
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64 
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --partition=general
#SBATCH --time=08:00:00


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env

DATASET_TYPE=${1:-""}
DATA_PATH=${2:-""}
MASTER_PORT=${3:-29500}
LEARNING_RATE=${4:-5e-7}
EXPERIMENT_DESC=${5:-""}
BASE_MODEL_PATH=${6:-""}

bash $REPO_DIR/scripts/slurm/babel_untested/gemma/train/train_dpo.sh $DATASET_TYPE $DATA_PATH $MASTER_PORT $LEARNING_RATE $EXPERIMENT_DESC $BASE_MODEL_PATH