#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate redteam

cd /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/data_generation
python evaluate_multiturn_attacks.py chat_completion=llama-3.1-8b-instruct.yaml multiturn_generated_attack_prompts_fname=/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/gpt-3.5-turbo-0125_advbench_generated_multiturn_prompts_20-10-1722471013.json
python evaluate_multiturn_attacks.py chat_completion=llama-3.1-8b-instruct.yaml multiturn_generated_attack_prompts_fname=/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/gpt-3.5-turbo-0125_openai_generated_multiturn_prompts_20-09-1722470963.json
python evaluate_multiturn_attacks.py chat_completion=llama-3.1-8b-instruct.yaml multiturn_generated_attack_prompts_fname=/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/gpt-3.5-turbo-0125_harmbench_generated_multiturn_prompts_20-10-1722471036.json