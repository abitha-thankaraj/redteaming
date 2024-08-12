#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

# Define the base path for the scripts
BASEPATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts"
LOGPATH="$BASEPATH/logs/mistral_eval"


# Execute the second Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=attacker_mistralai/Mistral-7B-Instruct-v0.1 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=untrained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ut_jbb_output.out" 2>&1


# Execute the fourth Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135" \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_t_jbb_output.out" 2>&1
