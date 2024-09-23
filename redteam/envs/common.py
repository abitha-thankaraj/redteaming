from gym.utils import seeding
from typing import Optional, Any

from redteam.inference.language_models import (
    LanguageModel,
    HuggingFaceLM,
    load_model_and_tokenizer,
)
from redteam.train.datasets import get_value_function_keywords


class GameConversation:
    def __init__(self, system_message: str = "", messages=None, value_function_keywords=None):
        self.messages = messages if messages is not None else []
        self.system_message = system_message
        self.value_function_keywords = (
            value_function_keywords
            if value_function_keywords is not None
            else get_all_value_function_keywords()
        )

    def set_system_message(self, system_message: str):
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        self.messages.append((role, message))

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def _format_messages(
        self, 
        system_role, 
        user_role, 
        assistant_role, 
        offset=0, 
        strip_user_messages=True,
        strip_assistant_messages=False
    ):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": system_role, "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[offset:]):
            if i % 2 == 0:
                if strip_user_messages:
                    msg = split_message(self.value_function_keywords, msg)
                ret.append({"role": user_role, "content": msg})
            else:
                if msg is not None:
                    if strip_assistant_messages:
                        msg = split_message(self.value_function_keywords, msg)
                    ret.append({"role": assistant_role, "content": msg})
        return ret

    def to_openai_api_messages(self, offset=0, strip_user_messages=True, strip_assistant_messages=False):
        """Convert the conversation to OpenAI chat completion format."""
        return self._format_messages(
            "system",
            "user",
            "assistant",
            offset=offset,
            strip_user_messages=strip_user_messages,
            strip_assistant_messages=strip_assistant_messages,
        )

    def to_game_message(self):
        """Convert the conversation to a game message."""
        ret = []
        for role, msg in self.messages:
            ret.append({"role": role, "content": msg})
        return ret

    def to_attacker_message(self):
        return self.to_openai_api_messages(offset=0, strip_user_messages=True)

    def to_defender_message(self):
        return self.to_openai_api_messages(offset=1, strip_user_messages=True)

    def to_judge_input(self):
        return self.to_openai_api_messages(offset=1, strip_user_messages=True, strip_assistant_messages=True)

    @classmethod
    def parse_messages(self, message_dict):
        messages = []
        for message in message_dict:
            messages.append((message["role"], message["content"]))
        return messages


class Policy(HuggingFaceLM):
    def __init__(self, model_name, model, tokenizer, generation_kwargs):
        super().__init__(model_name, model, tokenizer)
        self.generation_kwargs = generation_kwargs

    def act(self, obs):
        return self.generate(obs, **self.generation_kwargs)

    def act_batch(self, obses):
        raise NotImplementedError

    def reset(self):
        pass


def get_policy(config):
    model, tokeniser = load_model_and_tokenizer(
        config.model_name, config.model_dir, config.model_cache_dir, config.device
    )
    return Policy(config.model_name, model, tokeniser, config.generation_kwargs)


def get_all_value_function_keywords():
    value_function_map = get_value_function_keywords(value_function_type="all")
    value_function_keywords = set()
    for value_fn_type in value_function_map.keys():
        for k, v in value_function_map[value_fn_type].items():
            value_function_keywords.add(v)
    return value_function_keywords


def split_message(value_keywords, msg):
    for keyword in value_keywords:
        if msg.startswith(keyword):
            return msg[len(keyword) :].strip()
    return msg
