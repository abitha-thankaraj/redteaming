#!/bin/bash

# Starting the controller
python3 -m fastchat.serve.controller &

# Starting the model worker with the specified model path
python3 -m fastchat.serve.model_worker --model-path $DATA_DIR.cache/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873 &

# Starting the OpenAI API server on host 0.0.0.0 and port 8000
# python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8003

# Deploy the gradio interface to make it easier to type in the prompt
# TODO: Run this as background process?
python3 -m fastchat.serve.gradio_web_server  --host 0.0.0.0 --port 8001


# On your local machine: port forward; to babel
# ssh -N -L 8001:127.0.0.1:8001 <user_name>@<babel-node>
