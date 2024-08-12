#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

# Define the base path for the scripts
BASEPATH="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts"
LOGPATH="$BASEPATH/logs/llama_eval"

# Execute the first Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=attacker_meta-llama/Meta-Llama-3.1-8B-Instruct \
defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
experiment_desc=untrained_attacker_untrained_defender_mistral > "$LOGPATH/llama_ut_oai_output.out" 2>&1

# Execute the third Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135" \
defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/llama_t_oai_output.out" 2>&1


# Execute the second Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=attacker_meta-llama/Meta-Llama-3.1-8B-Instruct \
defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
experiment_desc=untrained_attacker_untrained_defender_mistral > "$LOGPATH/llama_ut_jbb_output.out" 2>&1


# Execute the fourth Python command
python "$BASEPATH/play_redteaming_game.py" \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_"$BASEPATH/logs/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135" \
defender.chat_completion_config.model=defender_meta-llama/Meta-Llama-3.1-8B-Instruct \
experiment_desc=trained_attacker_untrained_defender_mistral > "$LOGPATH/llama_t_jbb_output.out" 2>&1
