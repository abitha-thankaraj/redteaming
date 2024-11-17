#!/bin/bash


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env

# Command-line argument parsing

DATASET_TYPE=${1:-""}
DATA_PATH=${2:-""}
MASTER_PORT=${3:-29500}
LEARNING_RATE=${4:-5e-7}
EXPERIMENT_DESC=${5:-""}
BASE_MODEL_PATH=${6:-""}


MODEL_PATH="google/gemma-2-2b-it"
AGENT_TYPE="defender"


MAX_LENGTH=4096
RUN_NAME="multiturn_dpo_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"

# L40S
export NCCL_P2P_DISABLE=1

# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo "dpo" \
        --model_name_or_path $BASE_MODEL_PATH \
        --seed 42   \
        --data_path $DATA_PATH \
        --eval_data_path "" \
        --value_function_type "" \
        --value_function_experiment "" \
        --max_length $MAX_LENGTH \
        --agent_type $AGENT_TYPE \
        --dataset_type $DATASET_TYPE \
        --model_name $MODEL_PATH \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --exp_desc $EXPERIMENT_DESC \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3_cpu_offload.json \
        --bf16 True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1   \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "steps" \
        --eval_steps 100000  \
        --save_strategy "steps"  \
        --save_steps 100000  \
        --save_total_limit 1 \
        --learning_rate $LEARNING_RATE \
        --weight_decay 0.0   \
        --warmup_ratio 0.04   \
        --lr_scheduler_type "cosine"   \
        --logging_steps 1     \
        --tf32 True    \
        --model_max_length $MAX_LENGTH \
        --torch_empty_cache_steps 1 \
        --gradient_checkpointing True \
        --remove_unused_columns False


# Find the latest checkpoint directory inside the output path
# LATEST_CHECKPOINT=$(ls -td $LOGDIR/checkpoint-* | head -1)

LATEST_CHECKPOINT=$(ls -td "$LOGDIR"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in $LOGDIR. Exiting."
    exit 1
fi

DEFENDER_MODEL_DIR=$LATEST_CHECKPOINT
DEFENDER_MODEL_NAME="$EXPERIMENT_DESC"

# Launch evals
for temperature in 0.0 0.7 1.0
do
    sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_all.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
    # sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_oai.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
    # sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_jbb.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
    # sbatch $REPO_DIR/scripts/slurm/babel_untested/gemma/eval_bench/eval_gemma_ss.sh $temperature $DEFENDER_MODEL_DIR $DEFENDER_MODEL_NAME "$EXPERIMENT_DESC.temp$temperature"
done

