#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

# Define the base path for the scripts
BASEPATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts"
LOGPATH="$BASEPATH/logs/mistral_eval"

ATTACKER_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
ATTACKER_MODEL_NAME="attacker_${ATTACKER_MODEL_PATH}";
DEFENDER_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
DEFENDER_MODEL_NAME="defender_${DEFENDER_MODEL_PATH}";

# RWR 
TRAINED_ATTACKER_MODEL_PATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366"
TRAINED_ATTACKER_MODEL_NAME="rwr_trained_attacker_${TRAINED_ATTACKER_MODEL_PATH}";
TRAINED_DEFENDER_MODEL_PATH="/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_mistralai/Mistral-7B-Instruct-v0.1_2024-08-23-11-07-11-756/checkpoint-6"
TRAINED_DEFENDER_MODEL_NAME="rwr_trained_defender_${TRAINED_DEFENDER_MODEL_PATH}"
# SFT



# Execute the second Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=attacker_mistralai/Mistral-7B-Instruct-v0.1 \
# defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
# experiment_desc=untrained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ut_jbb_output.out" 2>&1


# # Execute the fourth Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135" \
# defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
# experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_t_jbb_output.out" 2>&1


# # Execute the fourth Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=rwr_trained_attacker_"$BASEPATH/logs/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366" \
# defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
# experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ta_rwr_utd_jbb_output.out" 2>&1


# # Execute the fourth Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=openai \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=rwr_trained_attacker_"$BASEPATH/logs/multiturn_rwr_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-19-17-16-11-631/checkpoint-366" \
# defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
# experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/mistral_ta_rwr_utd_oai_output.out" 2>&1


python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
experiment_desc=rwr_trained_attacker_trained_defender_mistral > "$LOGPATH/mistral_ta_rwr_td_oai_output.out" 2>&1


# Execute the fourth Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
experiment_desc=rwr_trained_attacker_trained_defender_mistral > "$LOGPATH/mistral_ta_rwr_td_jbb_output.out" 2>&1


