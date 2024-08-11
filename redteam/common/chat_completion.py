from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import openai
from openai import OpenAI
from fastchat.model import get_conversation_template
from typing import Optional, List, Dict

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
            openai.api_key = os.getenv("OAI_KEY")
            openai.base_url = None  # OpenAI API completion
        else:
            openai.api_key = "EMPTY"
            openai.base_url = config.url  # Local Fastchat server completion

        self.system_prompt = ""
        self._init_conversation()

    def _init_conversation(self):
        # self.conv = get_conversation_template(self.config.model)
        # Won't need this because everything is handled by OpenAI API server.
        self.conv = self.create_new_conversation()

    def create_new_conversation(self):
        return get_conversation_template("gpt-4")

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def process_conversation_history(
        self,
        use_special_tokens: bool,
    ) -> List[Dict[str, str]]:
        """
        Given the conversation list, processes it into appropriate format.

        Input:
            use_special_tokens (bool):
                whether to use special tokens while appending the multi-turn conversation or not
        
        Output:
            conversation history, which has the following format (List[Dict[str, str]]):
            [
                {
                    "role": "system",
                    "content": <system_prompt>,
                },
                {
                    "role": "user",
                    "content": <user_prompt>,
                },
                {
                    "role": "assistant",
                    "content": <model generation>,
                },
                ....
            ]

            if use_special_tokens is True, 
                it returns K turns of {"role": "user", "content": ...}, {"role": "assistant", "content": ...}

            if use_special_tokens is False,
                it converts the K turn conversation into a 1-turn one, 
                to remove the <use> </user> and <assistant> </assistant> tokens.
                It looks like the following:

                {
                    "role": "user",
                    "content": "User: <user_prompt>\n\n Assistant: <model_generation>\n\n User: ...."
                }

            Example Usage:
                conv_history = self.process_conversation_history(use_special_tokens=True)
        """
        conv_history = self.conv.to_openai_api_messages()

        if not use_special_tokens:
            # Extract the system prompt
            new_conv_history = [conv_history[0]]
            rolenames = {
                "user": "User: ",
                "assistant": "Assistant: ",
            }
            
            concatenated_messages = ""
            for index in range(1, len(conv_history)):
                concatenated_messages += rolenames[conv_history[index]["role"]]
                concatenated_messages += conv_history[index]["content"]
                concatenated_messages += "\n\n"
            
            new_conv_history.append(
                {"role": "user", "content": concatenated_messages}
            )

            conv_history = new_conv_history

        return conv_history

    def multiturn_chat_completion(
        self, 
        system_prompt: Optional[str] = None, 
        messages: List[str] = [],
    ) -> List[Dict[str, str]]:

        self._init_conversation()

        if system_prompt is not None:
            self.conv.set_system_message(system_prompt)
        elif self.system_prompt is not None:  # Preset for a batch of messages
            self.conv.set_system_message(self.system_prompt)

        for message in messages:
            self.conv.append_message(
                role="user", 
                message=message,
            )

            res = openai.chat.completions.create(
                model=self.config.model,
                messages=self.process_conversation_history(use_special_tokens=True),
                temperature=self.config.temperature,
            )

            self.conv.append_message(
                role="assistant", 
                message=res.choices[0].message.content,
            )

        return self.conv.to_openai_api_messages()
    
    def special_tokens_aware_multiturn_chat_completion(
        self,
        system_prompt: Optional[str] = None,
        messages: List[str] = [],
        use_special_tokens: bool = True,
    ) -> List[Dict[str, str]]:
        self._init_conversation()
        actual_conversation = self.create_new_conversation()

        if system_prompt is not None:
            self.conv.set_system_message(system_prompt)
            actual_conversation.set_system_message(system_prompt)
        elif self.system_prompt is not None: # Preset for a batch of messages
            self.conv.set_system_message(self.system_prompt)
            actual_conversation.set_system_message(self.system_prompt)

        for message in messages:
            self.conv.append_message(
                role="user", 
                message=message,
            )

            conv_history = self.process_conversation_history(
                use_special_tokens=use_special_tokens,
            )
            actual_conversation.append_message(
                role="user", 
                message=conv_history[-1]["content"],
            )

            res = openai.chat.completions.create(
                model=self.config.model,
                messages=conv_history,
                temperature=self.config.temperature,
            )

            self.conv.append_message(
                role="assistant", 
                message=res.choices[0].message.content,
            )
            actual_conversation.append_message(
                role="assistant",
                message=res.choices[0].message.content,
            )

        return {
            "conversation": self.conv.to_openai_api_messages(),
            "actual_conversation": actual_conversation.to_openai_api_messages(),
        }
        

    def reset(self):
        self._init_conversation()
    
    def single_turn_chat_completion(self, message: str, history: bool = False): 
        if history:
            self.conv.append_message(role="user", message=message)
        else:
            self.reset()
            self.conv.append_message(role="user", message=message)

        res = openai.chat.completions.create(
                model=self.config.model,
                messages=self.conv.to_openai_api_messages(),
                temperature=self.config.temperature,
            )

        self.conv.append_message(role="assistant", message=res.choices[0].message.content)

        return self.conv.to_openai_api_messages()

        
