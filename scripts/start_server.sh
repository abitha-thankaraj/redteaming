#!/bin/bash

#SBATCH --job-name=launch_server
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL


model="mistralai/Mistral-7B-Instruct-v0.1"
port=8003

cd /data/tir/projects/tir7/user_data/athankar/redteaming/scripts
./deploy_api.sh $model $port
