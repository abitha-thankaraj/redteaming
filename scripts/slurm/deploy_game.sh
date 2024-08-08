#!/bin/bash

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam

export HF_HOME="/data/tir/projects/tir6/bisk/athankar/projects/.cache";
export LOGDIR="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/$(date +'%Y-%m-%d-%H-%M-%S-%3N')";

# Starting the controller
python3 -m fastchat.serve.controller --host 0.0.0.0 --port $1 &
# Starting the model worker with the specified model path

# Attacker model
python3 -m fastchat.serve.model_worker --model-path $2 --host 0.0.0.0 --controller-address "http://localhost:$1" --port $4 --worker-address "http://localhost:$4"  --device=gpu:0 &
# Defender model
python3 -m fastchat.serve.model_worker --model-path $3 --host 0.0.0.0 --controller-address "http://localhost:$1" --port $5 --worker-address "http://localhost:$5" --device=gpu:1 &


python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port $6 --controller-address "http://localhost:$1" &

sleep 300; # Sleep for 300 seconds. Wait until the model is loaded
#TODO: Add a check to see if the model is loaded - python code?

# We use the server port as the chat completion port; becuase we access all the models through the server
python $7/redteaming/scripts/play_redteaming_game.py attacker.chat_completion.port=$6 defender.chat_completion.port=$6

# $1 controller port
# $2 attacker model path
# $3 defender model path
# $4 attacker worker port
# $5 defender worker port
# $6 api port
# $7 path to the script
