import transformers
import torch
from redteam.train.dataset_utils import read_json, RWRDatasetHelper
from redteam.train.common import get_tokenizer_separators
from redteam.train.datasets import MultiturnRWRDataset, MultiturnSFTDataset
from transformers.trainer_pt_utils import LabelSmoother
from redteam.utils.slack_me import slack_notification
from tqdm import tqdm

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
CACHE_DIR = "/data/tir/projects/tir6/bisk/athankar/projects/.cache"


def get_tokenizer(model_name):
    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        config = transformers.AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            model_max_length=20_000,
            padding_side="right",
            use_fast=False,
        )
        return tokenizer
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        config = transformers.AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            config=config,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            model_max_length=20_000,
            padding_side="right",
            use_fast=False,
        )
        return tokenizer

    else:
        raise Exception("Not implemented")


def stripped_decode(tokenizer, masked_tokens):
    return tokenizer.decode(
        torch.where(masked_tokens == IGNORE_TOKEN_ID, tokenizer.unk_token_id, masked_tokens)
    ).replace(tokenizer.unk_token, "")


if __name__ == "__main__":
    from redteam.train.sft import set_seed_everywhere

    set_seed_everywhere(42)
    # MODEL_NAMES = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.1"]
    MODEL_NAMES = ["mistralai/Mistral-7B-Instruct-v0.1"]

    # conversations = [c["conversation"] for c in read_json("dummy_conversations.json")]

    for model_name in MODEL_NAMES:
        print(f"Model: {model_name} | Outputs:")
        tokenizer, tokenizer_separators = get_tokenizer_separators(get_tokenizer(model_name))
        for data_dir in [
            "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat_length_added.json",
            "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_train_data_llama_rewards_flat.json",
            "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data_llama_rewards_flat.json",
        ]:

            # sft_dataset = MultiturnSFTDataset(
            #     conversations,
            #     tokenizer,
            #     tokenizer_separators,
            #     ignore_token_id=IGNORE_TOKEN_ID,
            # )

            eval_conversation_reward_dict = RWRDatasetHelper(
                data_dir,
                "defender",
                dataset_type="naive_balance",
                length_key="Mistral-7B-Instruct-v0.1_length",
                max_length=1024,
            ).get_conversations()

            rwr_dataset = MultiturnRWRDataset(
                eval_conversation_reward_dict["conversations"],
                tokenizer,
                tokenizer_separators,
                ignore_token_id=IGNORE_TOKEN_ID,
                reward_per_turns=eval_conversation_reward_dict["rewards"],
                gamma=0.9,
                min_reward=0.0,
            )
            from IPython import embed

            embed()

            # from IPython import embed; embed()
            # rwr_dataset = MultiturnRWRDataset(
            #     conversations,
            #     tokenizer,
            #     tokenizer_separators,
            #     ignore_token_id=IGNORE_TOKEN_ID,
            #     reward_per_turns=[[1., 2., 3.0]]*len(conversations),
            #     gamma=0.9,
            #     min_reward=0.0
            # )

            for i in tqdm(range(len(rwr_dataset))):
                # print(stripped_decode(tokenizer, rwr_dataset[i]["labels"]))
                # this fails if the last turn has 0 rewards. fix later
                # assert torch.where(rwr_dataset[i]["rewards"])[0].equal(torch.where(rwr_dataset[i]["labels"]!=IGNORE_TOKEN_ID)[0]), "Only non-masked tokens should have rewards"
                assert set(torch.where(rwr_dataset[i]["rewards"])[0].tolist()).issubset(
                    set(torch.where(rwr_dataset[i]["labels"] != IGNORE_TOKEN_ID)[0].tolist())
                ), "Only non-masked tokens should have rewards"

            print(f"Passed conversation level rewards test for {model_name}| {data_dir}")
            slack_notification(
                f"Passed conversation level rewards test for {model_name}| {data_dir}"
            )

    # '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
