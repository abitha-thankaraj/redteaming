#!/bin/bash
#SBATCH --job-name=high_contrast_best_of_n
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/slurm/%A_%a.err
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24 
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --array=0-9

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env


python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/best_of_n_generate.py \
                chunk=${SLURM_ARRAY_TASK_ID}