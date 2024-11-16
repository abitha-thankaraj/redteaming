#!/bin/bash


source ~/.bashrc
source /scratch/bcgv/athankaraj/redteaming/scripts/slurm/env_files/.delta_env

# sbatch $REPO_DIR/scripts/slurm/delta/qwen/logits/precompute_logits_for_file.sh /scratch/bcgv/datasets/redteaming/best_of_n/qwen/combined/combined_qwen_best_of_n_unlabelled.json
# sbatch $REPO_DIR/scripts/slurm/delta/qwen/logits/precompute_logits_for_file.sh /scratch/bcgv/datasets/redteaming/best_of_n/qwen/combined/combined_qwen_best_of_n_value_labeled.json
sbatch $REPO_DIR/scripts/slurm/delta/qwen/logits/precompute_logits_for_file.sh /scratch/bcgv/datasets/redteaming/best_of_n/qwen/combined/combined_qwen_no_best_of_n_unlabelled.json
sbatch $REPO_DIR/scripts/slurm/delta/qwen/logits/precompute_logits_for_file.sh /scratch/bcgv/datasets/redteaming/best_of_n/qwen/combined/combined_qwen_no_best_of_n_value_labeled.json