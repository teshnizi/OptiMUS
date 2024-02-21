from typing import Dict, Optional, Union, List
from openai import Client, OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class Agent:
    def __init__(self, name, description, client, llm="gpt-3.5-turbo", **kwargs):
        self.name = name
        self.description = description
        self.client = client
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.llm = llm

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List] = None,
        seed: int = 10,
    ) -> str:
        model = self.llm
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if not prompt is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        if type(self.client) in [OpenAI, Client]:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
            )
            content = completion.choices[0].message.content

        elif type(self.client) == MistralClient:
            messages = [
                ChatMessage(role=message["role"], content=message["content"])
                for message in messages
            ]
            completion = self.client.chat(
                model=model,
                messages=messages,
            )
            content = completion.choices[0].message.content

        return content

    def generate_reply(
        self,
        task: str,
        state: Dict,
        sender: "Agent",
    ) -> (str, Dict):
        return (
            "This is a reply from the agent. REPLY NOT IMPLEMENTED! Terminate the whole process!",
            state,
        )
