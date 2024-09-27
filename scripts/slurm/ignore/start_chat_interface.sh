#!/bin/bash

#SBATCH --job-name=mixtral-8x7b-instruct
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:4
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL



cd /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/slurm
./deploy_chat_interface.sh