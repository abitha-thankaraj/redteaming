# https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py
import logging
from typing import List, Dict
import torch
import gc, os, time
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)



def load_model_and_tokenizer(model_name, model_dir, model_cache_dir, device):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
    tokenizer.padding_side = "left"

    # Llama3 models don't have pad token
    if (
        tokenizer.pad_token_id is None
        and model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ):
        tokenizer.pad_token_id = 128004

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
    
    def generate(self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float= 1.0,
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
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

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

    def generate(self, conv: List[Dict], max_n_tokens: int = None, temperature: float = 0.7,top_p: float = 1.0):
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
                num_return_sequences=1, # https://github.com/huggingface/blog/blob/main/how-to-generate.md
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                top_p=1,
                temperature=1,  # To prevent warning messages
            )
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if "meta-llama/Meta-Llama-3.1-8B-Instruct" in self.model_name:
            outputs = outputs.split("assistant\n\n")[-1]
            
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

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


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
