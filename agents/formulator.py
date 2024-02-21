from typing import Dict
from agents.agent import Agent
import json
import openai

fix_prompt_template = """     
You are a mathematical formulator working with a team of optimization experts. The objective is to tackle a complex optimization problem, and your role is to fix a previously modelled {target}.

Recall that the {target} you modelled was

-----
{constraint}
-----

and your formulation you provided was

-----
{formulation}
-----

The error message is 

-----
{error}
-----

Here are the variables you have so far defined:

-----
{variables}
-----

Here are the parameters of the problem

-----
{parameters}
-----

Your task is carefully inspect the old {target} and fix it when you find it actually wrong.
After fixing it modify the formulation. Please return the fixed JSON string for the formulation.

The current JSON is 

-----
{json}
-----

"""

prompt_template = """
You are an expert mathematical formulator and an optimization professor at a top university. Your task is to model {targetType} of the problem in the standard LP or MILP form. 

Here is a {targetType} we need you to model:
-----
{targetDescription}
-----

Here is some context on the problem:
-----
{background}
-----

Here is the list of available variables:
-----
{variables}
-----

And finally, here is list of input parameters:
-----
{parameters}
-----

First, take a deep breath and explain how we should define the {targetType}. Feel free to define new variables if you think it is necessary. Then, generate a json file accordingly with the following format (STICK TO THIS FORMAT!):


{{
    "{targetType}": {{
      "description": "The description of the {targetType}",
      "formulation": "The LaTeX mathematical expression representing the formulation of the {targetType}"
    }},
    "auxiliary_constraints": [
        {{
            "description": "The description of the auxiliary constraint",
            "formulation": "The LaTeX mathematical expression representing the formulation of the auxiliary constraint"
        }}
    ]
    "new_variables": [
        {{
            "definition": "The definition of the variable",
            "symbol": "The symbol for the variable",
            "shape": [ "symbol1", "symbol2", ... ]
        }}
    ],
    
}}

- Your formulation should be in LaTeX mathematical format (do not include the $ symbols). 
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use '```json' and '```' to enclose the json file.
- Important: You can not define new parameters. You can only define new variables.Use CamelCase and full words for new variable symbols, and do not include indices in the symbol (e.g. ItemsSold instead of itemsSold or items_sold or ItemsSold_i)
- Use \\textup{{}} when writing variable and parameter names. For example (\\sum_{{i=1}}^{{N}} \\textup{{ItemsSold}}_{{i}} instead of \\sum_{{i=1}}^{{N}} ItemsSold_{{i}})
- Use \\quad for spaces.
- Use empty list ([]) if no new variables are defined.
- Always use non-strict inequalities (e.g. \\leq instead of <), even if the constraint is strict.
- Define auxiliary constraints when necessary. Set it to an empty list ([]) if no auxiliary constraints are needed. If new auxiliary constraints need new variables, add them to the "new_variables" list too.

Take a deep breath and solve the problem step by step.

"""


class Formulator(Agent):
    def __init__(self, client: openai.Client, **kwargs):
        super().__init__(
            name="Formulator",
            description="This is a mathematical formulator agent that is an expert in mathematical and optimization modeling and can define and modify variables, constraints, and objective functions. Does 3 things: 1) Defining constraints, variables, and objective functions, 2) Modifying constraints, variables, and objective functions, 3) Other things related to mathematical formulation. If the history is empty, start from this agent.",
            client=client,
            **kwargs,
        )

    def _formulate(self, target_type: str, state):
        for target in state[target_type]:
            if target["status"] == "not_formulated":
                self._create_new_formulation(target, target_type, state)
            elif target["status"] == "runtime_error":
                self._fix_available_formulation(target, target_type, state)
            elif target["status"] == "formulated":
                continue
            else:
                error_msg = f"Invalid status: {json.dumps(target, indent=4)}"
                raise RuntimeError(error_msg)

        return

    def _create_new_formulation(self, target, target_type: str, state):
        print(f"Formulating {target_type} ...")
        context = {}
        context[target_type] = {}
        prompt = prompt_template.format(
            background=state["background"],
            targetType=target_type,
            targetDescription=target["description"],
            variables=json.dumps(
                [
                    {
                        "definition": v["definition"],
                        "symbol": v["symbol"],
                        "shape": v["shape"],
                    }
                    for v in state["variables"]
                ],
                indent=4,
            ),
            parameters=json.dumps(
                [
                    {
                        "definition": p["definition"],
                        "symbol": p["symbol"],
                        "shape": p["shape"],
                    }
                    for p in state["parameters"]
                ],
                indent=4,
            ),
        )

        cnt = 3
        while cnt > 0:
            cnt -= 1
            try:
                # output = self.llm_call(messages=messages)
                response = self.llm_call(prompt=prompt, seed=cnt)
                print("=" * 10)
                print(response)
                print("=" * 10)
                output = response
                # delete until the first '```json'
                if "```json" in output:
                    output = output[output.find("```json") + 7 :]
                    output = output[: output.rfind("```")]

                # go back until the last character is a }
                while output[-1] != "}":
                    output = output[:-1]

                # go forward until the first character is a {
                while output[0] != "{":
                    output = output[1:]

                # if there are '$' in the output, remove them
                if "$" in output:
                    output = output.replace("$", "")

                # find "formulation": " in output
                formulation_start = output.find('"formulation"')
                # find "new_variables": " in output
                auxiliary_constraints_start = output.find('"auxiliary_constraints"')
                # go back until you find a closed bracket
                while output[auxiliary_constraints_start] != "}":
                    auxiliary_constraints_start -= 1
                while output[auxiliary_constraints_start] != '"':
                    auxiliary_constraints_start -= 1

                # extract the formulation
                formulation = output[
                    formulation_start + 16 : auxiliary_constraints_start
                ]
                # remove it from the output
                output = (
                    output[: formulation_start + 16]
                    + output[auxiliary_constraints_start:]
                )

                formulation = formulation.replace("\\\\", "\\")
                output = output.replace("\\", "\\\\")
                output = output.replace("\\", "\\\\")
                output = output.replace("\\\\quad", "\\\\")

                update = json.loads(output)
                update[target_type]["formulation"] = formulation

                # Extract related variables and parameters from the formulation
                related_stuff = self.get_related_stuff(
                    state, formulation, update["new_variables"]
                )
                update["variables_mentioned"] = related_stuff["variables_mentioned"]
                update["parameters_mentioned"] = related_stuff["parameters_mentioned"]

                if not "new_variables" in update.keys():
                    raise Exception("new_variables is not in the json file!")
                if not "formulation" in update[target_type].keys():
                    raise Exception("formulation is not in the json file!")
                if not "auxiliary_constraints" in update.keys():
                    update["auxiliary_constraints"] = []

                break
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print("=" * 10)
                print(e)
                print("=" * 10)
                print(prompt)
                print("=" * 10)
                print(response)
                print("=" * 10)
                print(
                    f"Invalid json format in {target_type} formulation!\n{e}\n Try again ..."
                )

        if cnt == 0:
            raise Exception("Invalid json format!")

        all_variable_symbols = [variable["symbol"] for variable in state["variables"]]

        for variable in update["new_variables"]:
            if variable["symbol"] in all_variable_symbols:
                # raise Exception(f"Variable {variable['symbol']} already exists!")
                continue
            else:
                variable["status"] = "formulated"
                state["variables"].append(variable)
                if not variable["symbol"] in update["variables_mentioned"]:
                    update["variables_mentioned"].append(variable["symbol"])

        target["formulation"] = update[target_type]["formulation"]
        target["status"] = "formulated"

        target["related_variables"] = update["variables_mentioned"]
        target["related_parameters"] = update["parameters_mentioned"]

        # Add auxiliary constraints
        if "auxiliary_constraints" in update.keys():
            for constraint in update["auxiliary_constraints"]:
                constraint["status"] = "formulated"

                constraint["formulation"] = (
                    constraint["formulation"]
                    .replace("\\\\", "\\")
                    .replace("\\\\", "\\")
                )

                # Extract related variables and parameters from the formulation
                related_stuff = self.get_related_stuff(
                    state, constraint["formulation"], []
                )
                constraint["related_variables"] = related_stuff["variables_mentioned"]
                constraint["related_parameters"] = related_stuff["parameters_mentioned"]

                state["constraint"].append(constraint)

        print(f"Formulation: {target['formulation']}")
        print("---")
        return

    def _fix_available_formulation(self, target, target_type: str, state):
        print(f"Fixing {target_type} ...")
        context = {}
        context[target_type] = {}
        context[target_type]["description"] = target["description"]

        context["variables"] = state["variables"]
        context["parameters"] = state["parameters"]
        context["formulation"] = target["formulation"]
        context["error"] = state["error_message"]

        prompt = fix_prompt_template.format(
            target=target_type,
            constraint=json.dumps(context[target_type]["description"], indent=4),
            variables=json.dumps(context["variables"], indent=4),
            parameters=json.dumps(context["parameters"], indent=4),
            formulation=json.dumps(context["formulation"], indent=4),
            json=json.dumps(target),
            error=context["error"],
        )

        cnt = 3
        while cnt > 0:
            cnt -= 1
            try:
                output = self.llm_call(prompt=prompt)
                # delete until the first '```json'
                output = output[output.find("```json") + 7 :]
                # delete until the last '```'
                output = output[: output.rfind("```")]

                output = output.replace(" \\", " \\\\")
                update = json.loads(output)

                break
            except Exception as e:
                print(e)
                print(
                    f"Invalid json format in {target_type} formulation! Try again ..."
                )

        if cnt == 0:
            raise Exception("Invalid json format!")

        target["formulation"] = update["formulation"]
        target["related_variables"] = update["related_variables"]
        target["related_parameters"] = update["related_parameters"]
        target["status"] = "formulated"

        return

    def generate_reply(self, task: str, state: Dict, sender: Agent) -> (str, Dict):
        # add some lines and characters around it to make the input interface nicer

        print("- Formulator agent is called!")
        print()

        self._formulate("constraint", state)
        self._formulate("objective", state)

        return "Formulation Done! Now we can write the code.", state

    def get_related_stuff(self, state, formulation, new_variables):
        ret = {}
        ret["variables_mentioned"] = []
        ret["parameters_mentioned"] = []

        # find all symbols enclosed in \textup{} in the formulation
        symbols_mentioned = []
        for i in range(len(formulation)):
            if formulation[i : i + 8] == "\\textup{":
                j = i + 8
                while formulation[j] != "}":
                    j += 1
                symbols_mentioned.append(formulation[i + 8 : j])

        all_parameter_symbols = [
            parameter["symbol"] for parameter in state["parameters"]
        ]
        all_variable_symbols = [variable["symbol"] for variable in state["variables"]]
        all_variable_symbols += [variable["symbol"] for variable in new_variables]

        # print(all_parameter_symbols)
        # print(all_variable_symbols)

        for symbol in symbols_mentioned:
            if symbol in all_parameter_symbols:
                ret["parameters_mentioned"].append(symbol)
            elif symbol in all_variable_symbols:
                ret["variables_mentioned"].append(symbol)
            elif symbol.lower().strip() in [
                "min",
                "max",
                "subject to",
                "s.t.",
                "st",
                "minimize",
                "maximize",
                "sum",
                "for all",
                "forall",
                "such that",
                "and",
                "or",
                "if",
                "then",
                "else",
                "otherwise",
                "for each",
                "exists",
                "foreach",
            ]:
                continue
            elif " " in symbol:
                continue
            else:
                raise Exception(f"Symbol {symbol} is not defined!")

        return ret
