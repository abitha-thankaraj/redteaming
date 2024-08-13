from typing import List, Dict
from copy import deepcopy


class ConversationTemplate:
    """
        Class to keep track of conversation with a model.
        This is used to keep track of the conversation in the same format as GPT-4.

        Input:
            roles (List[str]):
                Allowed roles in the conversation. 
                Default: ['system', 'user', 'assistant']
            
            system_prompt (str):
                The system prompt for the conversation.
                Default: "You are a helpful assistant."

        Example Usage:
            conv = ConversationTemplate()
            conv.append_message(
                role="user",
                message="Who are you?",
            )

            conv_history = conv.to_openai_api_messages()
            print(conv)
            # [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who are you?"}]
    """
    def __init__(
        self,
        roles: List[str] = ['system', 'user', 'assistant'],
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.roles = roles
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        """
            Resets the conversation to only contain the system prompt.
        """
        self.conv_history = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def set_system_message(
        self,
        system_prompt: str,
    ) -> None:
        """
        Sets the system prompt of the conversation
        to the given value.

        Input:
            system_prompt (str):
                The system prompt that we want to set.

        Output:
            None
        """
        self.system_prompt = system_prompt
        self.conv_history[0]["content"] = self.system_prompt

    def append_message(
        self, 
        role: str, 
        message: str,
    ) -> None:
        """
            Given a role and a message, it appends the corresponding message
            to the conversation history.

            Input:
                role (str):
                    The role of the given message. Has to be in the registered list
                    of roles
                
                message (str):
                    The message that needs to be saved

            Output:
                None
        """
        assert role in self.roles
        curr_conv = {
            "role": role,
            "content": message,
        }

        self.conv_history.append(curr_conv)

    def to_openai_api_messages(self) -> List[Dict[str, str]]:
        """
        Return the conversation in a List format.

        Input:
            None

        Output:
            conversation history, which has the following format:
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
                    "content": <model_generation>,
                },
                ...
            ]

        Usage:
            conv = ConversationTemplate()
            conv_history = conv.to_openai_api_messages()
        """
        return self.conv_history
    
    def process_conversation_history_without_special_tokens(
        self,
    ) -> List[Dict[str, str]]:
        """
        Given the conversation list, processes it into appropriate format.

        Input:
            None
        
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
            ]

            it converts the K turn conversation into a 1-turn one, 
            to remove the <use> </user> and <assistant> </assistant> tokens.
            It looks like the following:

            {
                "role": "user",
                "content": "User: <user_prompt>\n\n Assistant: <model_generation>\n\n User: ...."
            }
        """
        conv_history = deepcopy(self.conv_history)

        # Extract the system prompt
        if conv_history[0]["role"] == "system":
            new_conv_history = [conv_history[0]]
            conv_history = conv_history[1:]
        else:
            new_conv_history = []

        rolenames = {
            "user": "User: ",
            "assistant": "Assistant: ",
        }
            
        concatenated_messages = ""
        for index in range(len(conv_history)):
            concatenated_messages += rolenames[conv_history[index]["role"]]
            concatenated_messages += conv_history[index]["content"]
            concatenated_messages += "\n\n"
            
        new_conv_history.append(
            {"role": "user", "content": concatenated_messages}
        )

        return new_conv_history

