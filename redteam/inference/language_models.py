# https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py
import logging
from typing import List, Dict
import torch
import gc, os, time
import openai

logger = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def _init_conversations(self, goals: List[str]):
        """
        Initializes conversations for the language model.
        """
        raise NotImplementedError


# from vllm.entrypoints.llm import LLM

# class VLLMLanguageModel(LanguageModel):
#     def __init__(self, model_name, model, tokenizer):

#         self.model_name = model_name
#         self.model = model
#         if tokenizer.padding_side != "left":
#             logger.warning(
#                 "Padding side is not left. We use decoder only model. Setting padding side to left for inference"
#             )
#             tokenizer.padding_side = "left"
        
#         self.tokenizer = tokenizer
#         self.llm = LLM(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             tokenizer_mode="slow",
#             trust_remote_code=True,
#             dtype=torch.bfloat16,  # We train in half precision. We should also do inference?
#             seed=42,
#             # TODO: 
#             max_seq_len_to_capture=tokenizer.max_model_length,
#         )


#     def batched_generate(self, convs: List[List[Dict]], generation_config: Dict):
#         # Apply chat template to each prompt?
#         inputs = {}
#         inputs["input_ids"] = self.tokenizer.apply_chat_template(
#             convs, return_tensors="pt", padding=True, add_generation_prompt=True
#         )
#         inputs["attention_mask"] = inputs["input_ids"].ne(self.tokenizer.pad_token_id).long()
#         inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}




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
        ], "Only Meta-Llama and Mistral models are supported"
        # TODO: Assert that padding side is left for generations.
        assert (
            self.tokenizer.padding_side == "left"
        ), "Padding side must be left for generations for decoder only models"

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        num_samples: int = 1,
    ):
        # Apply chat template to each prompt?
        inputs = {}
        # TODO: Figure out if we need to do max length padding. does not seem necessary.
        inputs["input_ids"] = self.tokenizer.apply_chat_template(
            convs, return_tensors="pt", padding=True, add_generation_prompt=True
        )
        inputs["attention_mask"] = inputs["input_ids"].ne(self.tokenizer.pad_token_id).long()
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        max_n_tokens = min(max_n_tokens, self.model.config.max_length)

        #TODO : Add caching for the model generation: https://huggingface.co/docs/transformers/kv_cache#iterative-generation-with-cache
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_samples, # https://github.com/huggingface/blog/blob/main/how-to-generate.md
            )
        else:
            if num_samples > 1:
                logger.warning(
                    "num_samples > 1 is not supported for greedy decoding. Setting num_samples to 1."
                )
                num_samples = 1
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )

        # print(self.tokenizer.batch_decode(output_ids, skip_special_tokens=False))
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        # Batch decoding
        # print(self.tokenizer.batch_decode(output_ids, skip_special_tokens=False))
        # from IPython import embed; embed()
        print(output_ids)
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        if "meta-llama/Meta-Llama-3.1-8B-Instruct" in self.model_name:
            outputs_list = [output.replace("assistant\n\n", "") for output in outputs_list]

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list


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

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
