#!/bin/bash


source /home/athankar/.bashrc
source /home/athankar/miniconda3/etc/profile.d/conda.sh
module load cuda-12.3
conda activate redteam
source /data/user_data/athankar/redteaming/scripts/slurm/env_files/.babel_env



# Command-line argument parsing
MODEL_PATH=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
AGENT_TYPE=${2:-"defender"}
MASTER_PORT=${3:-29500}
DATASET_TYPE=${4:-"high_contrast"}
VALUE_FUNCTION_TYPE=${5:-""}
VALUE_FUNCTION_EXPERIMENT=${6:-""}
LEARNING_RATE=${7:-5e-7}
RWR_TEMPERATURE=${8:-1.0}
LENGTH_KEY=${9:-"Meta-Llama-3.1-8B-Instruct_length"}
EXPERIMENT_DESC=${10:-"dpo_min_test_high_contrast"}


MAX_LENGTH=4096
RUN_NAME="multiturn_dpo_${AGENT_TYPE}_${MODEL_PATH}_$(date +'%Y-%m-%d-%H-%M-%S-%3N')"
LOGDIR="$MODEL_PARENT_DIR/$RUN_NAME"


# Run the first job
deepspeed --master_port $MASTER_PORT $REPO_DIR/redteam/train/train.py  \
        --algo "dpo" \
        --model_name_or_path $MODEL_PATH \
        --seed 42   \
        --data_path $DATA_DIR/precomputed_datasets/meta-llama/Meta-Llama-3.1-8B-Instruct_defender_high_contrast_precomputed_logits_1000.pt \
        --eval_data_path $DATA_DIR/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat_length_added.json \
        --agent_type $AGENT_TYPE \
        --dataset_type $DATASET_TYPE \
        --length_key $LENGTH_KEY \
        --max_length $MAX_LENGTH \
        --value_function_type "$VALUE_FUNCTION_TYPE" \
        --model_name $MODEL_PATH \
        --value_function_experiment "$VALUE_FUNCTION_EXPERIMENT" \
        --rwr_temperature $RWR_TEMPERATURE \
        --rwr_type "" \
        --output_dir $LOGDIR  \
        --cache_dir $HF_HOME \
        --run_name $RUN_NAME \
        --exp_desc $EXPERIMENT_DESC \
        --deepspeed $REPO_DIR/scripts/configs/deepspeed/zero3.json \
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
LATEST_CHECKPOINT=$(ls -td $LOGDIR/checkpoint-* | head -1)
# Convert the latest checkpoint to fp32
python $LATEST_CHECKPOINT/zero_to_fp32.py -d $LATEST_CHECKPOINT $LATEST_CHECKPOINT/pytorch_model.bin


# # Schedule the evals to run after the first job; 0, 0.7, 1.0 temperature

# RWR_DEFENDER_MODEL_NAME="rwr_trained_defender"
# # Model parent dir is the parent directory for the checkpoint folder 
# # The checkpoint folder is where the model is loaded from
# RWR_DEFENDER_MODEL_PARENT_DIR=$LATEST_CHECKPOINT

# SFT_ATTACKER_MODEL_TYPE="sft_trained_attacker"
# SFT_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturnsft_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-10-16-56-06-894/checkpoint-135
# RWR_ATTACKER_MODEL_TYPE="rwr_trained_attacker"
# RWR_ATTACKER_MODEL_DIR=$MODEL_PARENT_DIR/multiturn_rwr_attacker_meta-llama/Meta-Llama-3.1-8B-Instruct_2024-08-23-13-23-23-840/checkpoint-183



# # Args: $DEFENDER_MODEL_PARENT_DIR $TEMPERATURE $DEFENDER_MODEL_NAME $ATTACKER_MODEL_DIR $ATTACKER_MODEL_NAME
# # for loop through temperatures
# for temperature in 0.0 0.7 1.0
# do
#     sbatch --dependency=afterok:$SLURM_JOB_ID $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $RWR_DEFENDER_MODEL_PARENT_DIR $RWR_DEFENDER_MODEL_NAME $SFT_ATTACKER_MODEL_DIR $SFT_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC
#     sbatch --dependency=afterok:$SLURM_JOB_ID $REPO_DIR/scripts/slurm/delta/evaluate_iter_0.sh $temperature $RWR_DEFENDER_MODEL_PARENT_DIR $RWR_DEFENDER_MODEL_NAME $RWR_ATTACKER_MODEL_DIR $RWR_ATTACKER_MODEL_TYPE $EXPERIMENT_DESC
# done
