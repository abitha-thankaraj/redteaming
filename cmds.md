```
bash /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/slurm/deploy_game_.sh 21042 mistralai/Mistral-7B-Instruct-v0.1 meta-llama/Meta-Llama-3.1-8B-Instruct 21043 21044 8031 /data/tir/projects/tir7/user_data/athankar/redteaming/scripts


<!-- python fastchat/serve/shutdown_serve.py --down all -->

python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py attacker.chat_completion.port=21033 attacker.model=attacker defender.chat_completion.port=21034 defender.model=defender


python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=attacker_mistralai/Mistral-7B-Instruct-v0.1 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=untrained_attacker_untrained_defender_mistral

python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=attacker_mistralai/Mistral-7B-Instruct-v0.1 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=untrained_attacker_untrained_defender_mistral


python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral


python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py \
dataset_configs=jailbreakbench \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral > output.log 2>&1


curl -X POST 0.0.0.0:22012/list_models
python /data/tir/projects/tir7/user_data/athankar/FastChat/fastchat/serve/shutdown_serve.py --down all


```

python3 -m fastchat.serve.openai_api_server \
    --host 0.0.0.0 \
    --port 8031 \
    --controller-address "http://localhost:21042"


```
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 20444 &
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker \
    --model-path "mistralai/Mistral-7B-Instruct-v0.1" \
    --host 0.0.0.0 \
    --model-name "attacker" \
    --controller-address "http://0.0.0.0:20444" \
    --port 20445 \
    --worker-address "http://0.0.0.0:20445" \
    --device cuda &
python3 -m fastchat.serve.register_worker --controller http://0.0.0.0:20444 --worker-name http://0.0.0.0:20445 

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker \
    --model-path "mistralai/Mistral-7B-Instruct-v0.1" \
    --host 0.0.0.0 \
    --model-name="defender" \
    --controller-address "http://localhost:20444" \
    --port 20446 \
    --worker-address "http://localhost:20446" \
    --device cuda &

python3 -m fastchat.serve.register_worker --controller http://0.0.0.0:20444 --worker-name http://0.0.0.0:20446


python3 -m fastchat.serve.openai_api_server \
    --host 0.0.0.0 \
    --port 8003 \
    --controller-address "http://localhost:20444"

```


| INFO | controller | Register a new worker: http://localhost:20445
2024-08-06 16:38:20 | ERROR | controller | Get status fails: http://localhost:20445, HTTPConnectionPool(host='localhost', port=20445): Max retries exceeded with url: /worker_get_status (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fd49d6b2230>: Failed to establish a new connection: [Errno 111] Connection refused'))

python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 21001 --controller-address "http://0.0.0.0:21001"



torchrun --nproc_per_node=4 --master_port=23122 /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/train/multiturn_sft.py \
    --bf16 True \
    --output_dir /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --remove_unused_columns False \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config '{"fsdp_transformer_layer_cls_to_wrap":"MistralDecoderLayer","fsdp_cpu_offload":false,"activation_checkpointing":true}' \    
    --deepspeed "/data/tir/projects/tir7/user_data/athankar/redteaming/redteam/configs/deepspeed_configs.yaml"


    --fsdp "full_shard auto_wrap" \
    --fsdp_config '{"fsdp_transformer_layer_cls_to_wrap":"MistralDecoderLayer","fsdp_cpu_offload":false,"activation_checkpointing":true}' \



torchrun --nproc_per_node=4 --master_port=6969 /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/train/multiturn_sft.py \
    --bf16 True \
    --output_dir /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --remove_unused_columns False \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True


deepspeed --master_port 32079 /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/train/multiturn_sft.py \
    --deepspeed /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/deepspeed/zero3.json \
    --bf16 True \
    --output_dir /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 100000 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --lazy_preprocess False
    
     \
    --report_to wandb \
    # > /home/yuxiaoq/RISE/log/log/taco/${task}.log 2>&1 &




python /data/tir/projects/tir7/user_data/athankar/redteaming/scripts/play_redteaming_game.py \
dataset_configs=openai \
oai_server_port=6911 \
attacker.chat_completion_config.model=trained_attacker_/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/logs/multiturnsft_attacker_mistralai/Mistral-7B-Instruct-v0.1_2024-08-10-16-11-32-379/checkpoint-135 \
defender.chat_completion_config.model=defender_mistralai/Mistral-7B-Instruct-v0.1 \
experiment_desc=trained_attacker_untrained_defender_mistral