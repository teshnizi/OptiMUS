from typing import Dict
from agents.agent import Agent
import json
import openai


class UserProxy(Agent):
    def __init__(self, client: openai.Client, **kwargs):
        super().__init__(
            name="UserProxy",
            description="This is a user proxy agent that interacts with the user to collect information about the problem.",
            client=client,
            **kwargs
        )
        self.prompt_template = """
You're a user proxy in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to interact with the user to collect information about the problem.

Here's the task that the team needs your help with:
-----
{task}
-----
Write a question to the user to collect the information needed to complete the task. Only generate the question, and don't generate any other text.
"""

    def generate_reply(self, task: str, state: Dict, sender: Agent) -> (str, Dict):
        # add some lines and characters around it to make the input interface nicer

        prompt = self.prompt_template.format(
            task=task,
        )
        messages = [
            {"role": "system", "content": prompt},
        ]

        question = self.llm_call(messages=messages)
        question = "\n" + "=" * 20 + "\n" + question + "\n" + "-" * 10 + "\n"

        response = input(question)

        messages += [
            {"role": "user", "content": response},
            {
                "role": "system",
                "content": "Now that you have the response, please write a statement to the group manager, describing the information you collected from the user. Only generate the statement, and don't generate any other text.",
            },
        ]

        information = self.llm_call(messages=messages)

        if not "user_clarifications" in state:
            state["user_clarifications"] = []
        state["user_clarifications"].append(information)

        return information, state
