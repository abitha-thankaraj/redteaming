from dataclasses import dataclass
import requests
import json


@dataclass
class ChatCompletionConfig:
    url: str = "http://localhost:8003/v1/chat/completions"
    model: str = "Mistral-7B-Instruct-v0.2"




class ChatCompletion:
    def __init__(self, config: ChatCompletionConfig):
        self.config = config

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
            return response.json()
        else:
            return {"error": response.status_code, "message": response.text}

if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello! What is your name?"}]
    config = ChatCompletionConfig()

    chat_completion = ChatCompletion(config)
    response = chat_completion.call_chat_completion(messages)
    print(response)