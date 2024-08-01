#!/bin/bash

#SBATCH --job-name=llama
#SBATCH --output=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.out
#SBATCH --error=/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/slurm/%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=athankar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL


model="meta-llama/Meta-Llama-3.1-8B-Instruct"
port=8003

cd /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/slurm
./deploy_api.sh $model $port
sleep 500

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate redteam
python /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/data_generation/evaluate_multiturn_attacks.py
