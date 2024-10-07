# Borrowed from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import os

# Add this here for wandb project name
os.environ["WANDB_ENTITY"] = "mt_redteam"
os.environ["WANDB_PROJECT"] = "redteaming"

from redteam.train import *

# get_dataset, get_trainer, ModelArguments, DataArguments, RWRArguments, TrainingArguments
import transformers
import math, torch, pathlib
from redteam.train.common import (
    set_seed_everywhere,
    safe_save_model_for_hf_trainer,
    get_tokenizer_separators,
)


def get_model_and_tokenizer(model_args, data_args, training_args):
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    # Set RoPE scaling factor - useful when you want to train with a larger context window
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # Tie the weights - Commonly used for memory efficient training?
    model.tie_weights()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # Changed to use the model name instead of the model path, because it will help in iterative training
        # Tokeniser is never modified -> Always use the default tokeniser
        data_args.model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Add padding token to the tokenizer + Add masking values. tokenizer separator will be used for masking in dataloader
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)
    # NOTE: if the token_id exceed the vocab_size will cause failing in training process! we need add special config and resize the embedding size!
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, tokenizer_separator


def train():
    global local_rank
    # Parse args; All configs sent to the model
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, RWRArguments, TrainingArguments)
    )
    model_args, data_args, rwr_args, training_args = parser.parse_args_into_dataclasses()

    # Set rwr args
    rwr_args.r_max = data_args.gamma**2 + data_args.gamma + 1
    rwr_args.r_min = 0.0

    training_args.data_args = data_args
    training_args.model_args = model_args
    training_args.rwr_args = rwr_args

    assert (
        training_args.model_max_length >= data_args.max_length
    ), "Model max length should be greater than data max length"

    # Seed everything
    set_seed_everywhere(training_args.seed)

    # Mostly for debugging
    local_rank = training_args.local_rank

    model, tokenizer, tokenizer_separator = get_model_and_tokenizer(
        model_args, data_args, training_args
    )

    train_dataset, eval_dataset = get_dataset(
        algo=training_args.algo,
        data_args=data_args,
        tokenizer=tokenizer,
        tokenizer_separator=tokenizer_separator,
    )

    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, training_args)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        if hasattr(train_dataset, "reward_to_gos"):
            unique, counts = np.unique(
                train_dataset.reward_to_gos.flatten(), return_counts=True
            )
            value_counts = dict(zip(unique, counts))
        else:
            value_counts = None
        training_args.dataset_stats = DatasetStats(
            train_dataset_length=len(train_dataset), rtg_dict=value_counts
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
