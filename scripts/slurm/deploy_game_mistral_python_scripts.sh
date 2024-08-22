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


# Execute the fourth Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=rwr_trained_attacker_"$BASEPATH/logs/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366" \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ta_rwr_utd_jbb_output.out" 2>&1


# Execute the fourth Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=rwr_trained_attacker_"$BASEPATH/logs/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366" \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ta_rwr_utd_oai_output.out" 2>&1