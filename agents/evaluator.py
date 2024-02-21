from typing import Dict
from agents.agent import Agent
import json
import openai
import traceback

main_prompt_templates = [
    """
You're an expert evaluator in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to run the code and evaluate the performance and correctness of the code.
"""
]


prep_code = """
import json
import numpy as np
import math

{solver_prep_code}

with open("{data_json_path}", "r") as f:
    data = json.load(f)

"""


post_code = """

# Get model status
status = model.status

obj_val = None
# check whether the model is infeasible, has infinite solutions, or has an optimal solution
if status == gp.GRB.INFEASIBLE:
    obj_val = "infeasible"
elif status == gp.GRB.INF_OR_UNBD:
    obj_val = "infeasible or unbounded"
elif status == gp.GRB.UNBOUNDED:
    obj_val = "unbounded"
elif status == gp.GRB.OPTIMAL:
    obj_val = model.objVal
"""


class Evaluator(Agent):
    def __init__(self, client: openai.Client, solver="gurobipy", **kwargs):
        super().__init__(
            name="Evaluator",
            description="This is an evaluator agent that is an expert in running optimization codes, identifying the bugs and errors, ane evaluating the performance and correctness of the code.",
            client=client,
            **kwargs,
        )
        self.solver = solver

    def generate_reply(self, task: str, state: Dict, sender: Agent) -> (str, Dict):
        print("- Evaluator agent is called!")

        res = self._run_code(state=state)

        if not res["success"]:
            state["solution_status"] = "runtime_error"
            state["error_message"] = res["error_message"]
            state["prep_code"] = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )

            if not res["bogus_context"]:
                return f"Bad model! Print DONE to finish the execution.", state

            res["bogus_context"]["status"] = "runtime_error"
            state["solver_output_status"] = res["bogus_context"]["status"]

            return (
                f"There was an error in running the code! {res['error_message']}",
                state,
            )

        else:
            state["solution_status"] = "solved"
            state["solver_output_status"] = res["status"]
            state["obj_val"] = res["obj_val"]
            state["code"] = res["code"]
            return ("Evaluation Done! The problem is solved.", state)

    def _run_code(self, state: Dict):
        local_env = {}
        code = ""
        last_line = ""
        bogus_context = None

        try:
            last_line = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )
            code += last_line + "\n"

            exec(
                last_line,
                local_env,
                local_env,
            )

            for parameter in state["parameters"]:
                if not "code" in parameter:
                    raise Exception(f"Parameter {parameter} is not coded yet!")
                last_line = parameter["code"]
                code += last_line + "\n"
                exec(last_line, local_env, local_env)

            # last_line = f"\n# Define model\nmodel = gp.Model('model')\n"
            # code += last_line + "\n"
            # exec(last_line, local_env, local_env)

            for variable in state["variables"]:
                bogus_context = variable
                last_line = variable["code"]
                code += last_line + "\n"
                exec(last_line, local_env, local_env)

            for constraint in state["constraint"]:
                bogus_context = constraint
                last_line = constraint["code"]
                code += "\n" + last_line + "\n"
                exec(last_line, local_env, local_env)

            bogus_context = state["objective"][0]
            last_line = state["objective"][0]["code"]
            code += "\n" + last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = "OPTIMIZATION CALL"
            last_line = f"\n# Optimize model\nmodel.optimize()\n"
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = None
            last_line = post_code
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            return {
                "success": True,
                "error_line": None,
                "code": code,
                "obj_val": local_env["obj_val"],
                "status": local_env["status"],
                "error_message": None,
            }

        except Exception as e:
            # print(local_env)
            print("COOOODE")
            print(code)
            print()
            if not bogus_context:
                error_msg = traceback.format_exc()
                raise Exception(
                    f"Unexpected error in running code at {last_line}: "
                    + "\n"
                    + str(e)
                    + "\n\n\n"
                    + error_msg
                )

            error_msg = traceback.format_exc()
            return {
                "success": False,
                "error_line": last_line,
                "code": code,
                "obj_val": None,
                "status": None,
                "error_message": error_msg,
                "bogus_context": bogus_context,
            }

    def get_solver_prep_code(self):
        if self.solver == "gurobipy":
            return "import gurobipy as gp\n\n # Define model\nmodel = gp.Model('model')"
        else:
            raise Exception(f"Solver {self.solver} is not supported yet!")
