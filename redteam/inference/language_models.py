# https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py
import logging
from typing import List, Dict
import torch
import gc, os, time
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, model_dir, model_cache_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_dir, use_fast=False
    )
    tokenizer.padding_side = "left"

    # Llama3 models don't have pad token
    if tokenizer.pad_token_id is None and model_name in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]:
        tokenizer.pad_token_id = 128004
    elif tokenizer.pad_token_id is None and model_name in [
        "mistralai/Mistral-7B-Instruct-v0.1"
    ]:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=model_cache_dir,
    ).to(device)
    model.tie_weights()
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


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
    ):
        """Generates a response for a prompt using a language model."""
        raise NotImplementedError


class HuggingFaceLM(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        # Decoder only models that have been verified
        assert model_name in [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ], "Only Meta-Llama and Mistral models are supported"
        # TODO: Assert that padding side is left for generations.
        assert (
            self.tokenizer.padding_side == "left"
        ), "Padding side must be left for generations for decoder only models"

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        decode_skip_special_tokens: bool = True,
    ):
        # Apply chat template to each prompt?
        inputs = {}
        inputs["input_ids"] = self.tokenizer.apply_chat_template(
            conv, return_tensors="pt", padding=True, add_generation_prompt=True
        )

        inputs["attention_mask"] = inputs["input_ids"].ne(self.tokenizer.pad_token_id).long()
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

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
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_timeout=self.API_TIMEOUT,
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output
