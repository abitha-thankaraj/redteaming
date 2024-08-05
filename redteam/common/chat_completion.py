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


class ChatCompletion(ABC):
    @abstractmethod
    def multiturn_chat_completion(self, messages: list[str]):
        pass


class OAIChatCompletion(ChatCompletion):
    def __init__(self, config) -> None:
        self.config = config
        if config.model.startswith("gpt"):
            openai.api_key = os.getenv("OAI_KEY")
            openai.base_url = None  # OpenAI API completion
        else:
            openai.api_key = "EMPTY"
            openai.base_url = config.url  # Local Fastchat server completion

        self.system_prompt = None
        self._init_conversation()

    def _init_conversation(self):
        self.conv = get_conversation_template(self.config.model)

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def multiturn_chat_completion(
        self, system_prompt=None, messages: list[str] = []
    ) -> list[str]:

        self._init_conversation()

        if system_prompt is not None:
            self.conv.set_system_message(system_prompt)
        elif self.system_prompt is not None:  # Preset for a batch of messages
            self.conv.set_system_message(self.system_prompt)

        for message in messages:
            self.conv.append_message(role="user", message=message)

            res = openai.chat.completions.create(
                model=self.config.model,
                messages=self.conv.to_openai_api_messages(),
                temperature=self.config.temperature,
            )

            self.conv.append_message(role="assistant", message=res.choices[0].message.content)

        return self.conv.to_openai_api_messages()
