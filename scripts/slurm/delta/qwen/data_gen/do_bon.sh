#!/bin/bash


source ~/.bashrc
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

for chunk in {0..9}
do
    sbatch $REPO_DIR/scripts/slurm/delta/qwen/data_gen/best_of_n.sh $chunk
    sbatch $REPO_DIR/scripts/slurm/delta/qwen/data_gen/best_of_n_.sh $chunk

done
