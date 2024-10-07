# https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py
from typing import List, Dict
import torch
import gc, os, time
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from redteam.train.common import get_tokenizer_separators


def load_model_and_tokenizer(
    model_name, model_dir, model_cache_dir, device, model_max_length=4096
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        use_fast=False,
        model_max_length=model_max_length,
    )
    tokenizer.padding_side = "left"
    tokenizer, tokenizer_separator = get_tokenizer_separators(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=model_cache_dir,
    ).to(device)
    model.tie_weights()
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, tokenizer_separator


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        decode_skip_special_tokens: bool = True,
        prefill_text: str = None,
    ):
        """Generates a response for a prompt using a language model."""
        raise NotImplementedError


class HuggingFaceLM(LanguageModel):
    def __init__(self, model_name, model, tokenizer, tokenizer_separator=None):
        super().__init__(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_separator = tokenizer_separator
        # Decoder only models that have been verified
        assert model_name in [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ], "Only Meta-Llama and Mistral models have been tested."
        # TODO: Assert that padding side is left for generations.
        if self.tokenizer.padding_side != "left":
            print(
                "Padding side is not left for the tokenizer. \
                Padding side must be left for generations for decoder only models. \
                Setting it to left."
            )
            self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        decode_skip_special_tokens: bool = True,
        prefill_text: str = None,
    ):
        """
        Given a conversation and a model, generates a response. We use only ancestral sampling for generation.
        Note - This function is only for instruction tuned/ chat tuned models.

        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
            decode_skip_special_tokens: bool, whether to skip special tokens
            prefill_text: str, prefix to start the generation with
        """
        # Apply chat template to each prompt.
        inputs = {}
        inputs["input_ids"] = self.tokenizer.apply_chat_template(
            conv, return_tensors="pt", padding=True, add_generation_prompt=True
        )
        # Prefilling the response.
        if prefill_text is not None:
            # The generation prompt is already added in the chat template.
            # This allows the model to start generating with a prefix.
            prefill_input_ids = self.tokenizer.encode(
                prefill_text, add_special_tokens=False, return_tensors="pt"
            )
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefill_input_ids], dim=-1)

        inputs["attention_mask"] = inputs["input_ids"].ne(self.tokenizer.pad_token_id).long()
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        # Greedy decoding
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,  # https://github.com/huggingface/blog/blob/main/how-to-generate.md
            )
        else:
            # Greedy decoding for temperature 0.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )
        outputs = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=decode_skip_special_tokens
        )
        #TODO: do this with tokenizer separators.
        if (
            "meta-llama/Meta-Llama-3.1-8B-Instruct" in self.model_name
            or "meta-llama/Meta-Llama-3-8B-Instruct" in self.model_name
        ):
            if decode_skip_special_tokens:
                outputs = outputs.split("assistant\n\n")[-1]
            else:
                # We use this when we use reserved tokens for value function experiments.
                outputs = outputs.split("<|start_header_id|>assistant<|end_header_id|>")[
                    -1
                ].split("<|eot_id|>")[0]

        if "mistralai/Mistral-7B-Instruct-v0.1" in self.model_name:
            outputs = outputs.split("[/INST] ")[-1]
        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids

        gc.collect()
        torch.cuda.empty_cache()
        return outputs

    @torch.no_grad()
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given input_ids, labels and attention mask,
        calculates the log probs (per token) from the auto-regressive generation process
        Args:
            input_ids: torch.Tensor, input_ids for the model
            labels: torch.Tensor, labels for the model
            attention_mask: torch.Tensor, causal attention_mask for the model
        """
        inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        model_outputs = self.model(**inputs)
        
        # run inference, get logits
        logits = (
            model_outputs["logits"]
            if isinstance(model_outputs, dict)
            else model_outputs[0]
        )

        # shift by 1, since we only consider causal models
        logits = logits[:, :-1, :]
        labels = labels[:, 1:].clone().to(self.model.device.index)

        # calculate log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # labels can contain ignore_token_id, typically -100.
        # we clamp them to 0, to avoid indexing errors
        # These tokens will be ignored later
        labels_clamped = torch.clamp(labels, min=0)
        log_probs = torch.gather(
            log_probs,
            dim=2,
            index=labels_clamped.unsqueeze(2),
        )

        for key in inputs:
            inputs[key].to("cpu")
        logits.to("cpu")

        log_probs = log_probs.to("cpu")

        del inputs, logits
        gc.collect()
        torch.cuda.empty_cache()

        return log_probs


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OAI_KEY")

    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output
