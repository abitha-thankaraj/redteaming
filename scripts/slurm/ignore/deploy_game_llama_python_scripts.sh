#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

# Define the base path for the scripts
BASEPATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts"
LOGPATH="$BASEPATH/logs/llama_eval"

# ATTACKER_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
# ATTACKER_MODEL_NAME="attacker_${ATTACKER_MODEL_PATH}";
DEFENDER_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFENDER_MODEL_NAME="defender_${DEFENDER_MODEL_PATH}";

#SFT
TRAINED_ATTACKER_MODEL_PATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135"
TRAINED_ATTACKER_MODEL_NAME="trained_attacker_${TRAINED_ATTACKER_MODEL_PATH}";

TRAINED_DEFENDER_MODEL_PATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturnsft_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-13-13-12-19-314/checkpoint-406";
TRAINED_DEFENDER_MODEL_NAME="trained_defender_${TRAINED_DEFENDER_MODEL_PATH}";

# RWR
# TRAINED_ATTACKER_MODEL_PATH="/data/group_data/rl/experiments/redteaming/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183"
# TRAINED_ATTACKER_MODEL_NAME="rwr_trained_attacker_${TRAINED_ATTACKER_MODEL_PATH}";

# TRAINED_DEFENDER_MODEL_PATH="/data/group_data/rl/experiments/redteaming/multiturn_rwr_defender_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-18-47-22-373/checkpoint-12"
# TRAINED_DEFENDER_MODEL_NAME="rwr_trained_defender_${TRAINED_DEFENDER_MODEL_PATH}";


# # Execute the first Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=openai \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=attacker_meta-llama/Meta-Llama-3.1-8B-Instruct \
# defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
# experiment_desc=untrained_attacker_untrained_defender_llama > "$LOGPATH/llama_ut_oai_output.out" 2>&1

# # Execute the third Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=openai \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135" \
# defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
# experiment_desc=trained_attacker_untrained_defender_llama > "$LOGPATH/llama_t_oai_output.out" 2>&1


# # Execute the second Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=attacker_meta-llama/Meta-Llama-3.1-8B-Instruct \
# defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
# experiment_desc=untrained_attacker_untrained_defender_llama > "$LOGPATH/llama_ut_jbb_output.out" 2>&1


# # Execute the fourth Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135" \
# defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
# experiment_desc=trained_attacker_untrained_defender_llama > "$LOGPATH/llama_t_jbb_output.out" 2>&1

# Execute the fourth Python command
# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
# defender.chat_completion_config.model="$DEFENDER_MODEL_NAME" \
# experiment_desc=rwr_trained_attacker_untrained_defender_llama > "$LOGPATH/llama_ta_rwr_utd_jbb_output.out" 2>&1


# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=openai \
# oai_server_port=6911 \
# attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
# defender.chat_completion_config.model="$DEFENDER_MODEL_NAME" \
# experiment_desc=rwr_trained_attacker_untrained_defender_llama > "$LOGPATH/llama_ta_rwr_utd_oai_output.out" 2>&1


# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=jailbreakbench \
# oai_server_port=6911 \
# attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
# defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
# experiment_desc=rwr_trained_attacker_trained_defender_llama > "$LOGPATH/llama_ta_rwr_td_jbb_output.out" 2>&1


# python "$BASEPATH/play_redteaming_game.py" \
# dataset_configs=openai \
# oai_server_port=6911 \
# attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
# defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
# experiment_desc=rwr_trained_attacker_trained_defender_llama > "$LOGPATH/llama_ta_rwr_td_oai_output.out" 2>&1

#ta-td-sft

python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
experiment_desc=sft_trained_attacker_trained_defender_llama > "$LOGPATH/llama_ta_sft_td_oai_output.out" 2>&1

python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model="$TRAINED_ATTACKER_MODEL_NAME" \
defender.chat_completion_config.model="$TRAINED_DEFENDER_MODEL_NAME" \
experiment_desc=sft_trained_attacker_trained_defender_llama > "$LOGPATH/llama_ta_sft_td_jbb_output.out" 2>&1