import os
import json
from utils import extract_list_from_end, get_response, extract_json_from_end

import re


directions = """

And here's how the solver is imported and set up:

...
from gurobipy import Model, GRB

model = Model("OptimizationProblem")
...

Use model.addConstr() to add the constraint to the model.
"""

prompt_constraints_code = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{description}

Your task is to write {solver} code for the following constraint in python:

{constraint}

Here's a list of parameters that are related to the constraint:

{params}

Here's a list of variables related to the constraint:

{vars}


{directions}

The code should be written in the following format:

CODE
=====
code for defining the constraint (ONLY the constraint definition code, without the imports, the variable definitions, and the solver setup)
=====
    
Here's an example for modeling $\\forall i, SalesVolumes[i] \leq MaxProductionVolumes[i]$ where shape of both SalesVolumes and MaxProductionVolumes is [N]:

CODE
=====
for i in range(N):
    model.addConstr(SalesVolumes[i] <= MaxProductionVolumes[i])
=====

- Do not generate anything after the last =====.
- Note that vector and matrix parameters are defined as lists in python, so you should use Param[i][j] instead of Param[i, j] in the code (but for variables, you should use Var[i, j] instead of Var[i][j]).
- Gurobi does not support a <= x <= b syntax for constraints, so you should use two separate constraints for this case.

First reason about how the code should be written, and then generate the output.
Take a deep breath and think step by step.
"""


prompt_objective_code = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{description}

Your task is to write {solver} code for the objective in python:

{objective}

Here's a list of parameters that are related to the objective:

{params}

Here's a list of variables related to the constraint:

{vars}

{directions}


CODE
=====
code for defining the objective (ONLY the objective definition code, without the imports, the variable definitions, and the solver setup)
=====
    
Here's an example for modeling $\\max \\sum_{{i=1}}^{{N}} price_i x_i$ where shape of both price and x is [N]:

CODE
=====
model.setObjective(quicksum(price[i] * x[i] for i in range(N)), GRB.MAXIMIZE)
=====

- Do not generate anything after the last =====.
- Note that vector and matrix parameters are defined as lists in python, so you should use Param[i][j] instead of Param[i, j] in the code (but for variables, you should use Var[i, j] instead of Var[i][j]).

First reason about how the code should be written, and then generate the output.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


qs = []


def extract_code_from_end(text):
    # get 1st and 2nd occurence of "====="
    if "=====" in text:
        ind_1 = text.find("=====")
        ind_2 = text.find("=====", ind_1 + 1)

        code = text[ind_1 + len("=====") : ind_2].strip()
    else:
        ind_1 = text.find("```python")
        ind_2 = text.find("```", ind_1 + 1)

        code = text[ind_1 + len("```") : ind_2].strip()

    if "```" in code:
        code = code.replace("```python", "").replace("```", "").strip()

    if code.startswith("====="):
        code = code[len("=====") :].strip()

    if code.endswith("====="):
        code = code[: -len("=====")].strip()

    if "python" in code:
        code = code.replace("python", "").strip()

    return code


def get_codes(
    desc,
    params,
    vars,
    constraints,
    objective,
    model,
    check=False,
):

    coded_constraints = []
    for c in constraints.copy():
        k = 1
        while k > 0:
            try:
                prompt = prompt_constraints_code.format(
                    solver="gurobipy",
                    description=desc,
                    params=json.dumps(params, indent=4),
                    vars=json.dumps(vars, indent=4),
                    constraint=json.dumps(c, indent=4),
                    directions=directions,
                )
                res = get_response(prompt, model=model)

                print("\n\n\n\n+++++")
                print(res)

                code = extract_code_from_end(res)

                print("+++++")
                print(code)

                c["code"] = code
                coded_constraints.append(c)
                break

            except Exception as e:
                k -= 1
                if k == 0:
                    raise (e)

    coded_objective = {
        "description": objective["description"],
        "formulation": objective["formulation"],
    }

    k = 1
    while k > 0:
        try:
            prompt = prompt_objective_code.format(
                solver="gurobipy",
                description=desc,
                params=json.dumps(params, indent=4),
                vars=json.dumps(vars, indent=4),
                objective=json.dumps(objective, indent=4),
                directions=directions,
            )
            res = get_response(prompt, model=model)
            print("\n\n\n\n+++++")
            print(res)
            print("+++++")

            code = extract_code_from_end(res)
            print(code)
            coded_objective["code"] = code
            break
        except Exception as e:
            k -= 1
            if k == 0:
                raise (e)

    return coded_constraints, coded_objective
