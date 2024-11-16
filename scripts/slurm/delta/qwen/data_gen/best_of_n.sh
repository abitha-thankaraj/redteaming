#!/bin/bash
#SBATCH --job-name=high_contrast_best_of_n
#SBATCH --output=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.err
#SBATCH --account=bcgv-delta-gpu
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuA40x4
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --exclude=gpub054

source ~/.bashrc
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
module load cuda/12.4.0
conda activate redteam

# Loads common environment variables
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

CHUNK=$1

python /scratch/bcgv/athankaraj/redteaming/scripts/best_of_n_generate.py chunk=$CHUNK

