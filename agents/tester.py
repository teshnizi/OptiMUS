from typing import Dict
from agents.agent import Agent
import json
import openai

test_code_template = """
You are in a team of optimization experts, and you are responsible for testing the validity of the solution
    obtained for the problem. Your job is to verify that all the constraints of the problem are satisfied by the 
    solution. Here is one constraint you need to verify 

    {CONSTRAINT}

    This constraint is built based on the following variables 

    {VARIABLE}

    Now the problem is solved and solution is obtained. 
    Please write Python code that gets values from the Python variables and then verify the constraint.
"""


class AutoTester(Agent):

    def __init__(self, client: openai.Client, solver="gurobipy", **kwargs):

        super(AutoTester, self).__init__(
            name="AutoTester", description="This is a testing agent that double checks the validity of the solution")

        self.solver = solver

    def generate_reply(
        self,
        task: str,
        state: Dict,
        sender: "Agent",
    ) -> (str, Dict):

        print("- Testing agent is called! ")





