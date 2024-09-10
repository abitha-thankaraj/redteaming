import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from dataclasses import dataclass, field


@dataclass
class RWRArguments:
    rwr_temperature: float = field(
        default=0.9,
        metadata={"help": "RWR temperature (β) . Higher -> | Lower -> "},
    )
    rwr_type: str = field(
        default="exp", metadata={"help": "RWR term type. Available options: exp"}
    )


# Offline RWR -  TODO: Test step by step.


class RWRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rwr_temperature = kwargs.get("rwr_temperature", 1.0)
        self.rwr_type = kwargs.get("rwr_type", "exp")

    def get_rwr_term(self, rewards, rwr_type):
        if rwr_type == "exp":
            return torch.exp(rewards / self.rwr_temperature)
        elif rwr_type == "baseline_batch_mean":  # batch size is 1?
            return (
                rewards - rewards.mean()
            )  # if this diverges, then go to something online with impportance clipping.
        # elif rwr_type == "baseline_ema_batch_mean":
        #     return rewards - self._ema_rewards.mean()
        # elif rwr_type == "baseline_running_mean":

        # elif rwr_type == "rloo":

        else:
            raise NotImplementedError(f"RWR type {rwr_type} not implemented")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Offline RWR loss function:
            L_RWR = - log(π(a|s)) * exp(r(s,a)/β)
        """

        labels = inputs["labels"]
        rewards = inputs.pop("rewards", torch.zeros_like(labels))  # shape should be (B, T)
        model_output = model(
            **inputs
        )  # Standard forward pass, for testing purposes we can use the loss term from here.

        if self.args.past_index >= 0:
            self._past = model_output[self.args.past_index]

        model_name = self.accelerator.unwrap_model(model)._get_name()

        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            shift_labels = True

        # Logits shape: (B, T, V); B - Batch size, T - Sequence length, V - Vocabulary size
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        # This is for loss computation with the
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            rewards = rewards[
                ..., 1:
            ].contiguous()  # Shift rewards to the same extent as the labels

        # This is just cross-entropy; but weighted by the RWR term
        log_probs = -F.log_softmax(logits, dim=-1)  # Log softmax over the vocabulary dimension
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(
                -1
            )  # Adds a dimension to make it broadcastable for the vocabulary dimension?
        if rewards.dim() == log_probs.dim() - 1:
            rewards = rewards.unsqueeze(
                -1
            )  # Adds a dimension to make it broadcastable for the vocabulary dimension?

        padding_mask = labels.eq(-100)  # B x T
        # Makes the -100 to 0;
        labels = torch.clamp(labels, min=0)
        # Gather the log probs at the label indices over the vocabulary dimension
        nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss.masked_fill_(padding_mask, 0.0)
        # Multiply by the RWR term; You dont calculate the loss on any of the masked terms.
        nll_loss = nll_loss * self.get_rwr_term(rewards, self.rwr_type)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        # Average over only the non-masked elements
        nll_loss = nll_loss.sum() / num_active_elements
        return (nll_loss, model_output) if return_outputs else nll_loss


"""
To write tests:
    - model_outputs["loss"] should be equal to the loss computed by the RWRTrainer when rewards = 0.
    - Wegihted loss - should change?

"""
