from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import openai
from openai import OpenAI
from fastchat.model import get_conversation_template

"""
    #OAI Usage
    oai_config = ChatCompletionConfig(model="gpt-3.5-turbo")
    oai_chat_completion = OAIChatCompletion(oai_config)
    oai_cc = oai_chat_completion.multiturn_chat_completion(system_prompt=None, messages=["hello", "have you heard of voldemort?", "is wizardry real?"])

    #Fastchat Usage
    config = ChatCompletionConfig()
    chat_completion = OAIChatCompletion(config)
    mistral_cc = chat_completion.multiturn_chat_completion(system_prompt=None, messages=["hello", "have you heard of voldemort?", "is wizardry real?"])

"""


@dataclass
class ChatCompletionConfig:
    port: int = 8003
    url: str = "http://localhost:8003/v1/"
    model: str = "Mistral-7B-Instruct-v0.1"
    temperature: float = 1.0
    n: int = 1
    max_tokens: int = 1024


class ChatCompletion(ABC):
    @abstractmethod
    def multiturn_chat_completion(self, messages: list[str]):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def single_turn_chat_completion(self, message: str):
        pass


class OAIChatCompletion(ChatCompletion):
    def __init__(self, config) -> None:
        self.config = config
        if config.model.startswith("gpt"):
            self.client = OpenAI(
                api_key=os.getenv("OAI_KEY"),
            )
        else:
            self.client = OpenAI(api_key="EMPTY", base_url=config.url)

        self.system_prompt = ""
        self._init_conversation()

    def _init_conversation(self):
        self.conv = get_conversation_template("gpt-4")
        # We dont want the default system prompt to be set by the conversation template
        self.conv.set_system_message(self.system_prompt)

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def multiturn_chat_completion(
        self, system_prompt=None, messages: list[str] = []
    ) -> list[str]:

        self._init_conversation()

        if system_prompt is not None:
            self.conv.set_system_message(system_prompt)

        for message in messages:
            self.conv.append_message(role="user", message=message)

            res = self.client.chat.completions.create(
                model=self.config.model,
                messages=self.conv.to_openai_api_messages(),
                temperature=self.config.temperature,
            )

            self.conv.append_message(role="assistant", message=res.choices[0].message.content)

        return self.conv.to_openai_api_messages()

    def reset(self):
        self._init_conversation()

    def single_turn_chat_completion(self, message: str, history: bool = False):
        if history:
            self.conv.append_message(role="user", message=message)
        else:
            self.reset()
            self.conv.append_message(role="user", message=message)

        res = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.conv.to_openai_api_messages(),
            temperature=self.config.temperature,
        )
        self.conv.append_message(role="assistant", message=res.choices[0].message.content)

        return self.conv.to_openai_api_messages()
