#!/bin/bash
#SBATCH --job-name="halos_dpo"
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=general
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=0
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH -t 48:00:00

cd /home/ftajwar
source .bashrc
conda activate fastchat
cd /home/ftajwar/redteaming/redteam


python data_generation/generate_multiturn_attack_prompts.py dataset_configs=harmbench


