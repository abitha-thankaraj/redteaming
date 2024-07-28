from dataclasses import dataclass
import requests
import json


@dataclass
class ChatCompletionConfig:
    url: str = "http://localhost:8003/v1/chat/completions"
    model: str = "Mistral-7B-Instruct-v0.1"


class ChatCompletion:
    def __init__(self, config: ChatCompletionConfig):
        self.config = config

    def create_message(self, prompt:str, role:str="user") -> dict:
        """Message formatting for chat completion API. 
        """
        return {"role": role, "content": prompt}

    def call_chat_completion(self, messages: list) -> dict:
        url = self.config.url
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.config.model,
            "messages": messages
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            response = response.json()
            return response["choices"][0]["message"]
        else:
            return {"error": response.status_code, "message": response.text}
    
    def multiturn_chat_completion(self, messages: list[str]) -> dict:
        conversation_history = []
        for message in messages:
            # user prompt
            conversation_history.append(self.create_message(message))
            # prompt = history
            response = self.call_chat_completion(conversation_history)
            # assistant response
            conversation_history.append(response)

        return conversation_history

# if __name__ == "__main__":
#     messages = [{"role": "user", "content": "Hello! What is your name?"}]
#     config = ChatCompletionConfig()

#     chat_completion = ChatCompletion(config)
#     # chat_completion.multiturn_chat_completion(["hello", "have you heard of voldemort?", "is wizardry real?"])

#     # response = chat_completion.call_chat_completion(messages)

#     prompts = ["hello", "have you heard of voldemort?", "is wizardry real?"]
#     conversation = chat_completion.multiturn_chat_completion(prompts)
#     print(conversation)
