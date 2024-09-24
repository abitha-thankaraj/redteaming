python /data/tir/projects/tir7/user_data/athankar/redteaming/redteam/train/train_rwr.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat_length_added.json" \
    --eval_data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json" \
    --agent_type defender \
    --dataset_type "weighted" \
    --value_function_experiment "prefix" \
    --value_function_type "natural_lang_multiclass" \
    --dataset_type_weights 0.5 0.3 0.2 \
    --num_samples 7814 \
    --length_key Meta-Llama-3.1-8B-Instruct_length \
    --max_length 4096 \
    --model_max_length 4096 \
    --output_dir /home/athankar

source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
conda activate redteam
module load cuda-12.3

# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# LENGTH_KEY="Mistral-7B-Instruct-v0.1_length"
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
LENGTH_KEY="Meta-Llama-3.1-8B-Instruct_length"

AGENT_TYPE="defender"

# "meta-llama/Meta-Llama-3.1-8B-Instruct"
MASTER_PORT=29500
MAX_LENGTH=4096
DATASET_TYPE="all"
VALUE_FUNCTION_TYPE=""
VALUE_FUNCTION_EXPERIMENT=""
# VALUE_FUNCTION_TYPE=$1
# VALUE_FUNCTION_EXPERIMENT=$2
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

LEARNING_RATE=1e-7

# "Meta-Llama-3.1-8B-Instruct_length"
# MODEL_PATH=$1
# AGENT_TYPE=$2
# MASTER_PORT=$3
#


# export HF_HOME="/data/group_data/rl/models"
#
export HF_TOKEN="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";

HF_HOME= "/data/tir/projects/tir6/bisk/athankar/projects/.cache";
# "/data/tir/projects/tir6/bisk/athankar/projects/.cache";
RUN_NAME="multiturn_rwr_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')";
LOGDIR="/data/group_data/rl/experiments/redteaming/$RUN_NAME";
REPO_DIR="/data/tir/projects/tir7/user_data/athankar/redteaming";
DATA_DIR="/data/group_data/rl/datasets/redteaming";


deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train_rwr.py  \
        --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat_length_added.json" \
        --eval_data_path "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json" \
        --agent_type defender \
        --dataset_type "weighted" \
        --value_function_experiment "prefix" \
        --value_function_type "natural_lang_multiclass" \
        --dataset_type_weights 0.5 0.3 0.2 \
        --num_samples 7814 \
        --length_key Meta-Llama-3.1-8B-Instruct_length \
        --max_length 4096 \
        --model_max_length 4096 \
        --model_name $MODEL_NAME \
        --rwr_temperature 1.0 \
        --rwr_type "raw_rewards" \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --experiment-desc raw_rewards \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json     \
        --bf16 True \
        --num_train_epochs 2  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "steps" \
        --eval_steps 100000  \
        --save_strategy "steps"  \
        --save_steps 100000  \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE  \
        --weight_decay 0.    \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length 4096 \
        --remove_unused_columns False \

