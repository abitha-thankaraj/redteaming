#!/bin/bash
#SBATCH --job-name=logits
#SBATCH --output=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.out
#SBATCH --error=/scratch/bcgv/athankaraj/logs/slurm/%A_%a.err
#SBATCH --account=bcgv-delta-gpu
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuA40x4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest 
#SBATCH --no-requeue
#SBATCH --time=04:30:00
#SBATCH --exclude=gpub054

source ~/.bashrc
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
module load cuda/12.4.0
conda activate redteam

# Loads common environment variables
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

python /scratch/bcgv/athankaraj/redteaming/scripts/data_generation/precompute_logits.py data_flat_fname=$1

