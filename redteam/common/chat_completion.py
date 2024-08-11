from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import openai
from copy import deepcopy
from redteam.common.conversation_template import ConversationTemplate
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

        if hasattr(self.config, 'system_prompt'):
            self.system_prompt = self.config.system_prompt
        else:
            self.system_prompt = ""
        self._init_conversation()

    def _init_conversation(self):
        self.conv = ConversationTemplate(
            system_prompt=self.system_prompt,
        )

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.conv.set_system_message(
            system_prompt=system_prompt,
        )

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
                messages=self.conv.to_openai_api_messages(),
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
    ) -> List[Dict]:
        """
        This function lets us evaluate the multi-turn attacks in two ways:

        Way 1: (use_special_tokens = True)
            This is the default option. In this case the conversation happens
            as usual. The model gets the following input (or its equivalent)
            and asked to generate the output:

            <system> system_prompt </system>
            <user> question 1 </user>
            <assistant> response from the model </assistant>
            .... (K - 1) previous turns
            <user> question K </user

        Way 2: (use special tokens = False)
            This is the alternate way. Here we format the K turn conversation
            into a 1-turn dialog without the special tokens.
            
            The model gets the following input and asked to generate:

            <system> system_prompt </system>
            <user>
            User: question 1
            Assistant: answer 1
            ....
            User: question k - 1
            Assistant: answer k - 1
            User: question K
            </user>

        Input:
            system_prompt (Optional[str]):
                The system prompt for this conversation
                Overrides any prior system prompt if set to None
                Otherwise does nothing

                Default: None

            messages (List[str]):
                The questions that needs to be asked, in sequence.

                Default: []

            use_special_tokens (bool):
                Whether to use special tokens or not.
                If True, uses Way 1 above
                If False, uses Way 2

                Default: True
        
        Output:
            A list of dictionary of the following format:
            {
                "conversation": <conversation>,
                "actual_conversation": <actual_conversation>,
            }

            <conversation> is the sequence of questions 
                and responses we want to evaluate.
            <actual_conversation> is the sequence of concatenated inputs
                for the model and outputs.
        """
        self._init_conversation()

        if system_prompt is not None:
            self.conv.set_system_message(system_prompt)
        elif self.system_prompt is not None: # Preset for a batch of messages
            self.conv.set_system_message(self.system_prompt)

        actual_conversation = deepcopy(self.conv.to_openai_api_messages())
        for message in messages:
            self.conv.append_message(
                role="user",
                message=message,
            )

            processed_conv_history = (
                self.conv.process_conversation_history_without_special_tokens()
                if not use_special_tokens
                else self.conv.to_openai_api_messages()
            )
            actual_conversation.append(processed_conv_history[-1])

            res = openai.chat.completions.create(
                model=self.config.model,
                messages=processed_conv_history,
                temperature=self.config.temperature,
            )

            self.conv.append_message(
                role="assistant", 
                message=res.choices[0].message.content,
            )
            actual_conversation.append(
                {
                    "role": "assistant",
                    "content": res.choices[0].message.content,
                }
            )

        return {
            "conversation": self.conv.to_openai_api_messages(),
            "actual_conversation": actual_conversation,
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