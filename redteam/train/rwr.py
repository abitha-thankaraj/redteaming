import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# Offline RWR

class RWRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = kwargs.get("temperature", 1.0)
        self.rwr_type = kwargs.get("rwr_type", "exp") # Default is 

    def get_rwr_term(self, rewards, rwr_type):
        if rwr_type == "exp":
            return torch.exp(rewards / self.temperature)
        else:
            raise NotImplementedError(f"RWR type {rwr_type} not implemented")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Offline RWR loss function:
            L_RWR = - log(π(a|s)) * exp(r(s,a)/β)
        """

        labels = inputs.pop("labels")
        # rewards shape is B x tokenizer_model_max_length; Same as the labels
        rewards = inputs.pop("rewards", torch.zeros_like(labels)) # shape should be (B,)
        model_output = model(**inputs)
        if self.args.past_index >= 0:
            self._past = model_output[self.args.past_index]
        
        model_name = self.accelerator.unwrap_model(model)._get_name()

        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            shift_labels = True
            
        # Logits shape: (B, T, V); B - Batch size, T - Sequence length, V - Vocabulary size
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            # What does this shift do? - Apparently next token sequence matching?
        log_probs = -F.log_softmax(logits, dim=-1) # Log softmax over the vocabulary dimension

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1) # Adds a dimension to make it broadcastable for the vocabulary dimension?
        
        padding_mask = labels.eq(self.label_smoother.ignore_index) # B x T
        # Makes the -100 to 0
        labels = torch.clamp(labels, min=0)
        # Gather ???? at dims when labels are non zero?
        nll_loss = log_probs.gather(dim=-1, index=labels) # Gather the log probs at the label indices over the vocabulary dimension
        
        nll_loss.masked_fill_(padding_mask, 0.0)
        # Modification to add RWR term??? padding_mask - can this be point multiplied with the rewards?
        reward_padding_mask = padding_mask * self.get_rwr_term(rewards, self.rwr_type)
        reward_padding_mask = reward_padding_mask.view(-1, 1) # Adds view to make it broadcastable for the vocabulary dimension.
        # Multiply by the RWR term
        nll_loss = nll_loss * reward_padding_mask
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        
        nll_loss = nll_loss.sum() / num_active_elements
        
        return (nll_loss, model_output) if return_outputs else nll_loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.

#         Subclass and override for custom behavior.
#         """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]

#         if labels is not None:
#             unwrapped_model = self.accelerator.unwrap_model(model)
#             if _is_peft_model(unwrapped_model):
#                 model_name = unwrapped_model.base_model.model._get_name()
#             else:
#                 model_name = unwrapped_model._get_name()
#             if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#         return (loss, outputs) if return_outputs else loss


# @dataclass
# class LabelSmoother:
#     """
#     Adds label-smoothing on a pre-computed output from a Transformers model.

#     Args:
#         epsilon (`float`, *optional*, defaults to 0.1):
#             The label smoothing factor.
#         ignore_index (`int`, *optional*, defaults to -100):
#             The index in the labels to ignore when computing the loss.
#     """

#     epsilon: float = 0.1
#     ignore_index: int = -100

#     def __call__(self, model_output, labels, shift_labels=False, rwr_term= None):
#         logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
#         # TODO: Add rewards to the model_output

#         if shift_labels:
#             logits = logits[..., :-1, :].contiguous()
#             labels = labels[..., 1:].contiguous()
#             # TODO What does this shift do?
#         log_probs = -nn.functional.log_softmax(logits, dim=-1)
#         # TODO: Multiply by e^(r/beta) - does this have to be different for per turn rewards?

#         if labels.dim() == log_probs.dim() - 1:
#             labels = labels.unsqueeze(-1)

#         padding_mask = labels.eq(self.ignore_index) * rwr_term
#         # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
#         # will ignore them in any case.
#         labels = torch.clamp(labels, min=0)
#         nll_loss = log_probs.gather(dim=-1, index=labels)
#         # TODO: If it is only at the end of a trajectory - Only multiply after gather.
#         # works for fp16 input tensor too, by internally upcasting it to fp32
#         smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

#         nll_loss.masked_fill_(padding_mask, 0.0)
#         smoothed_loss.masked_fill_(padding_mask, 0.0)

#         # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
#         num_active_elements = padding_mask.numel() - padding_mask.long().sum()
#         nll_loss = nll_loss.sum() / num_active_elements
#         smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
#         return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
