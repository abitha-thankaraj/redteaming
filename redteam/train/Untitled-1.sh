python /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/train/train_rwr.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat_length_added.json" \
    --eval_data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json" \
    --agent_type defender \
    --dataset_type "weighted" \
    --value_function_experiment "prefix" \
    --value_function_type "multilabel" \
    --dataset_type_weights 0.5 0.3 0.2 \
    --num_samples 7814 \
    --length_key Meta-Llama-3.1-8B-Instruct_length \
    --max_length 4096 \
    --model_max_length 4096 \
    --output_dir /home/athankar


