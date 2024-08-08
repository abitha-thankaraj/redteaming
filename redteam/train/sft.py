# Borrwed from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict
import transformers
import torch

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.1")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="/data/tir/projects/tir6/bisk/athankar/projects/.cache")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def preprocess(sources, tokenizer):
#     def preprocess(
#     sources, tokenizer: transformers.PreTrainedTokenizer, template_id, **kwargs
# ) -> Dict:
#     systems = None if not kwargs else kwargs.get("systems", None)

#     # If the data volume is small, process it directly in the main thread
#     if len(sources) <= 1000:
#         conversations, conv = apply_prompt_template(sources, template_id, systems)
#         input_ids, targets = tokenize_conversations(conversations, tokenizer)
#         targets = mask_targets(conversations, targets, tokenizer, conv)
#     else:  # If the data volume is large, use multithreading for processing
#         with Pool() as p:
#             conversations, conv = p.apply_async(
#                 apply_prompt_template, (sources, template_id, systems)
#             ).get()
#             input_ids, targets = p.apply_async(
#                 tokenize_conversations, (conversations, tokenizer)
#             ).get()
#             targets = p.apply_async(
#                 mask_targets, (conversations, targets, tokenizer, conv)
#             ).get()
#             p.close()
#             p.join()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

if __name__ == "__main__":
    main()