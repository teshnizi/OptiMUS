from typing import Dict
from agents.agent import Agent
import json
import openai


variable_definition_prompt_templates = [
    """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for defining variables of the problem.
""",
    """
Here's a variable we need you to write the code for defining:

-----
{variable}
-----

Assume the parameters are defined. Now generate a code accordingly and enclose it between "=====" lines. Only generate the code, and don't generate any other text. Here's an example:

**input**:

{{
    "definition": "Quantity of oil i bought in month m",
    "symbol": "buy_{{i,m}}",
    "shape": ["I","M"] 
}}

***output***:

=====
buy = model.addVars(I, M, vtype=gp.GRB.CONTINUOUS, name="buy")
=====


- Note that the indices in symbol (what comes after _) are not a part of the variable name in code.
- Use model.addVar instead of model.addVars if the variable is a scalar.

""",
]

main_prompt_templates = {
    "constraint": [
        """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for different constraints of the problem. 
""",
        """
Here's a constraint we need you to write the code for, along with the list of related variables and parameters:

-----
{context}
-----

- Assume the parameters and variables are defined, and gurobipy is imported as gp. Now generate a code accordingly and enclose it between "=====" lines. 
- Only generate the code and the ===== lines, and don't generate any other text.
- If the constraint requires changing a variable's integralilty, generate the code for changing the variable's integrality rather than defining the variable again.
- If there is no code needed, just generate the comment line (using # ) enclosed in ===== lines explaining why.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Here's an example:


**input**:


{{
    "description": "in month m, it is possible to store up to storageSize_{{m}} tons of each raw oil for use later.",
    "formulation": "\(storage_{{i,m}} \leq storageSize, \quad \\forall i, m\)",
    "related_variables": [
        {{
            "symbol": "storage_{{i,m}}",
            "definition": "quantity of oil i stored in month m",
            "shape": [
                "I",
                "M"
            ]
        }}
        ],
    "related_parameters": [
        {{
            "symbol": "storageSize_{{m}}",
            "definition": "storage size available in month m",
            "shape": [
                "M"
            ]
        }}
    ]
}}

***output***:

=====
# Add storage capacity constraints
for i in range(I):
    for m in range(M):
        model.addConstr(storage[i, m] <= storageSize[m], name="storage_capacity")
=====

Take a deep breath and approach this task methodically, step by step.

""",
    ],
    "objective": [
        """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for the objective function of the problem.
""",
        """
Here's the objective function we need you to write the code for, along with the list of related variables and parameters:

-----
{context}
-----

Assume the parameters and variables are defined, and gurobipy is imported as gp. Now generate a code accordingly and enclose it between "=====" lines. Only generate the code and the =====s, and don't generate any other text. Here's an example:

**input**:

{{
    "description": "Maximize the total profit from selling goods",
    "formulation": "Maximize \(Z = \sum_{{k=1}}^{{K}} \sum_{{i=1}}^{{I}} (profit_k \cdot x_{{k,i}} - storeCost \cdot s_{{k,i}})\)",
    "related_variables": [
        {{
            "symbol": "x_{{k,i}}",
            "definition": "quantity of product k produced in month i",
            "shape": [
                "K",
                "I"
            ],
            "code": "x = model.addVars(K, I, vtype=gp.GRB.CONTINUOUS, name='x')"
        }},
        {{
            "symbol": "s_{{k,i}}",
            "definition": "quantity of product k stored in month i",
            "shape": [
                "K",
                "I"
            ],
            "code": "s = model.addVars(K, I, vtype=gp.GRB.CONTINUOUS, name='s')"
        }}
    ],
    "related_parameters": [
        {{
            "symbol": "profit_{{k}}",
            "definition": "profit from selling product k",
            "shape": [
                "K"
            ]
        }},
        {{
            "symbol": "storeCost",
            "definition": "price of storing one unit of product",
            "shape": []
        }}
    ]
}}


***output***:

=====
# Set objective
m.setObjective(gp.quicksum(profit[k] * x[k, i] - storeCost * s[k, i] for k in range(K) for i in range(I)), gp.GRB.MAXIMIZE)
=====

Take a deep breath and approach this task methodically, step by step.

""",
    ],
}

debugging_prompt_templates = [
    """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for {target} of the problem.
""",
    """ 


When running a code snippet, an error happened. Here is the initial part of the code snippet for importing packages and defining the model:

-----
{prep_code}
-----

And here is the code for defining the related parameters and variables:

-----
{context}
-----

And the error happened when running this line:

-----
{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the import code is correct. First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",    
    "reason": ?
}}

Otherwise, fix the code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": ?
}}


- Note that the fixed code should be the fixed version of the original error line, not the whole code snippet.
- Do not generate any text after the json file. All the imports and model definition are already done, and you should only generate the fixed code to be replaced with the original error line.

""",
]

debugging_refined_template_target = """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for of the problem.

When running the following code snipper, an error happened:

-----
{prep_code}

{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the code for importing packages and defining parameters and variables is correct, and the error is because of the this last part which is for modeling the {target}:

-----
{error_line}
-----

First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",    
    "reason": "A string explaining why you think the problem is in the formulation"
}}

otherwise, fix the last part code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": "A sting representing the fixed {target} modeling code to be replaced with the last part code"
}}

- Note that the fixed code should be the fixed version of the last part code, not the whole code snippet. Only fix the part that is for modeling the {target}.
- Do not generate any text after the json file. 
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Take a deep breath and solve the problem step by step.

"""

debugging_refined_template_variable = """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for of the problem.

When running the following code snipper, an error happened:

-----
{prep_code}

{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the code for importing packages and defining parameters and variables is correct, and the error is because of the this last part which is for modeling the {target}:

-----
{error_line}
-----

First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",    
    "reason": "A string explaining why you think the problem is in the formulation"
}}

otherwise, fix the last part code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": "A sting representing the fixed {target} modeling code to be replaced with the last part code"
}}

- Note that the fixed code should be the fixed version of the last part code, not the whole code snippet. Only fix the part that is for defining the {target}.
- Do not generate any text after the json file. 
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Take a deep breath and solve the problem step by step.

"""


class Programmer(Agent):
    def __init__(
        self, client: openai.Client, solver="gurobipy", debugger_on=True, **kwargs
    ):
        super().__init__(
            name="Programmer",
            description="This is a mathematical programmer agent that is an expert in writing, modifying, and debugging code for optimization problems from the mathematical formulation of the problem. This agent should be called first when a bug or error happens in the code.",
            client=client,
            **kwargs,
        )

        self._debugger_on = debugger_on
        self.solver = solver

    def generate_reply(self, task: str, state: Dict, sender: Agent) -> (str, Dict):
        # add some lines and characters around it to make the input interface nicer
        print("- Programmer agent is called!")
        print()

        if state["solution_status"] == "runtime_error":
            # Enter debugging mode
            bogus_item = None
            for target in ["constraint", "objective", "variables"]:
                for item in state[target]:
                    if not item["status"] in ["coded", "formulated", "runtime_error"]:
                        # raise Exception(
                        #     f"{target} {item} inconsistency in state! \n {json.dumps(state, indent=4)}"
                        # )
                        print(
                            f"{target} {item} inconsistency in state! \n {json.dumps(state, indent=4)}"
                        )
                    if item["status"] == "runtime_error":
                        bogus_item = item
                        break

            if not bogus_item:
                raise Exception(
                    "No runtime error in state!", json.dumps(state, indent=4)
                )

            return self._debug_code(state=state)

        elif state["solution_status"] is None:
            # Enter coding mode
            return self._generate_code_from_formulation(state=state)

        else:
            raise Exception(
                f"Invalid solver_output_status {state['solver_output_status']}!"
            )

    def _debug_code(self, state: Dict) -> (str, Dict):
        if not self._debugger_on:
            raise Exception("Debugger is off. Execution failed")

        error_line = None
        bogus_context = None

        for target in ["constraint", "objective", "variables"]:
            for item in state[target]:
                if item["status"] == "runtime_error":
                    bogus_context = item

        context = {}
        prep_code = state["prep_code"]

        if "description" in bogus_context:
            error_line = bogus_context["code"]
            error_message = state["error_message"]
            for parameter in state["parameters"]:
                if parameter["symbol"] in bogus_context["related_parameters"]:
                    prep_code += parameter["code"] + "\n"

            for variable in state["variables"]:
                if variable["symbol"] in bogus_context["related_variables"]:
                    if not "code" in variable:
                        raise Exception(f"Variable {variable} is not coded yet!")

                    prep_code += variable["code"] + "\n"
            prompt = debugging_refined_template_target.format(
                target=target,
                prep_code=prep_code,
                error_line=error_line,
                error_message=error_message,
            )

        elif "definition" in bogus_context:
            # s = f"Debugging for Variable not implemented yet!\n {json.dumps(state, indent=4)}"
            # raise Exception(s)
            error_line = bogus_context["code"]
            error_message = state["error_message"]
            # for parameter in state["parameters"]:
            #     if parameter["symbol"] in bogus_context["related_parameters"]:
            #         prep_code += parameter["code"] + "\n"

            # for variable in state["variables"]:
            #     if variable["symbol"] in bogus_context["related_variables"]:
            #         if not "code" in variable:
            #             raise Exception(f"Variable {variable} is not coded yet!")

            #         prep_code += variable["code"] + "\n"

            prompt = debugging_refined_template_variable.format(
                target=target,
                prep_code=prep_code,
                error_line=error_line,
                error_message=error_message,
            )

        else:
            raise Exception(
                f"Invalid bogus_context {bogus_context}! \n {json.dumps(state, indent=4)}"
            )

        # messages = [
        #     {
        #         "role": "system",
        #         "content": debugging_prompt_templates[0].format(target=target),
        #     },
        #     {
        #         "role": "user",
        #         "content": debugging_prompt_templates[1].format(
        #             prep_code=state["prep_code"],
        #             error_line=error_line,
        #             error_message=error_message,
        #             context=json.dumps(context, indent=4),
        #         ),
        #     },
        # ]

        # print("%^%^%")
        # print(messages[1]["content"])

        cnt = 3
        while cnt > 0:
            cnt -= 1
            try:
                print("%^%^%")
                print(prompt)
                response = self.llm_call(prompt=prompt, seed=cnt)
                print(response)
                print("%^%^%")
                # delete until the first '```json'
                response = response[response.find("```json") + 7 :]
                # delete until the last '```'
                response = response[: response.rfind("```")]

                update = json.loads(response)

                if update["status"] == "correct":
                    bogus_context["status"] = "formulation_error"
                    return update["reason"], state
                elif update["status"] == "fixed":
                    bogus_context["status"] = "coded"
                    bogus_context["code"] = update["fixed_code"]
                    return "The code is fixed! Try evaluating it again.", state
                else:
                    raise Exception(f"Invalid status {update['status']}!")

            except Exception as e:
                print(e)
                print(f"Invalid json format {response}! Try again ...")

    def _generate_code_from_formulation(self, state: Dict) -> (str, Dict):
        for variable in state["variables"]:
            print(f"Programming variable {variable['symbol']} ...")

            if variable["status"] == "not_formulated":
                raise Exception(f"Variable {variable} is not formulated yet!")

            elif variable["status"] == "formulated":
                context = {}
                context["definition"] = variable["definition"]
                context["symbol"] = variable["symbol"]
                context["shape"] = variable["shape"]

                messages = [
                    {
                        "role": "system",
                        "content": variable_definition_prompt_templates[0].format(
                            solver=self.solver
                        ),
                    },
                    {
                        "role": "user",
                        "content": variable_definition_prompt_templates[1].format(
                            variable=context,
                        ),
                    },
                ]

                cnt = 3
                while cnt > 0:
                    try:
                        response = self.llm_call(messages=messages, seed=cnt)
                        print(response)
                        code = [
                            r.strip()
                            for r in response.split("=====")
                            if len(r.strip()) > 2
                        ][-1]

                        code = code.strip()
                        while code[0] == "=":
                            code = code[1:].strip()
                        while code[-1] == "=":
                            code = code[:-1].strip()

                        if len(code) < 2:
                            raise Exception(f"Invalid code {code}!")

                        code = code.replace("```python", "").replace("```", "")

                        variable["code"] = code
                        variable["status"] = "coded"
                        break
                    except Exception as e:
                        cnt -= 1
                        import traceback

                        print(traceback.print_exc())
                        print(messages[1]["content"])
                        print(response)
                        print(e)
                        print(f"Invalid response {response}! Try again ...")

                        if cnt == 0:
                            raise e

            elif variable["status"] == "coded":
                pass

        for target in ["constraint", "objective"]:
            for item in state[target]:
                print(f"Programming {target} ...")
                if item["status"] == "not_formulated":
                    raise Exception(f"{target} {item} is not formulated yet!")

                elif item["status"] == "formulated":
                    context = {}
                    context["description"] = item["description"]
                    context["formulation"] = item["formulation"]
                    context["related_variables"] = []
                    context["related_parameters"] = []

                    for parameter in state["parameters"]:
                        if parameter["symbol"] in item["related_parameters"]:
                            context["related_parameters"].append(parameter)

                    for variable in state["variables"]:
                        if variable["symbol"] in item["related_variables"]:
                            if not "code" in variable:
                                raise Exception(
                                    f"Variable {variable} is not coded yet!"
                                )
                            context["related_variables"].append(
                                {
                                    "symbol": variable["symbol"],
                                    "definition": variable["definition"],
                                    "shape": variable["shape"],
                                    "code": variable["code"],
                                }
                            )

                    messages = [
                        {
                            "role": "system",
                            "content": main_prompt_templates[target][0].format(
                                solver=self.solver
                            ),
                        },
                        {
                            "role": "user",
                            "content": main_prompt_templates[target][1].format(
                                context=json.dumps(context, indent=4),
                            ),
                        },
                    ]

                    cnt = 3

                    while cnt > 0:
                        try:
                            response = self.llm_call(messages=messages, seed=cnt)
                            print(response)
                            code = [
                                r.strip()
                                for r in response.split("=====")
                                if len(r.strip()) > 2
                            ][-1]

                            code = code.replace("```python", "").replace("```", "")

                            item["code"] = code
                            item["status"] = "coded"
                            break
                        except Exception as e:
                            import traceback

                            print(traceback.print_exc())
                            print(messages[1]["content"])
                            print(response)
                            cnt -= 1
                            if cnt == 0:
                                raise e

                else:
                    raise Exception(f"{target} {item} is not formulated yet!")

        return "Coding Done! Now we can evaluate the code!", state
