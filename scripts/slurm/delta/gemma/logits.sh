#!/bin/bash


source ~/.bashrc
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

sbatch $REPO_DIR/scripts/slurm/delta/gemma/get_logits.sh /scratch/bcgv/datasets/redteaming/best_of_n/gemma/combined/combined_gemma_best_of_n_unlabelled.json
sbatch $REPO_DIR/scripts/slurm/delta/gemma/get_logits.sh /scratch/bcgv/datasets/redteaming/best_of_n/gemma/combined/combined_gemma_best_of_n_value_labeled.json
sbatch $REPO_DIR/scripts/slurm/delta/gemma/get_logits.sh /scratch/bcgv/datasets/redteaming/best_of_n/gemma/combined/combined_gemma_no_best_of_n_unlabelled.json
sbatch $REPO_DIR/scripts/slurm/delta/gemma/get_logits.sh /scratch/bcgv/datasets/redteaming/best_of_n/gemma/combined/combined_gemma_no_best_of_n_value_labeled.json