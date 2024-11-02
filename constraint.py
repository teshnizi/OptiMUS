import json
import pandas as pd
from rag.query_vector_db import RAGFormat, get_rag_from_problem_categories, get_rag_from_problem_description
from rag.rag_utils import RAGMode, constraint_path
from utils import (
    extract_list_from_end,
    get_response,
    extract_json_from_end,
)

import re


prompt_constraints = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{rag}-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}


Your task is to identify and extract constraints from the description. The constraints are the conditions that must be satisfied by the variables. Please generate the output in the following python list format:

[
    Constraint 1,
    Constraint 2,
    ...
]

for example:
    
[
    "Sum of weights of all items taken should not exceed the maximum weight capacity of the knapsack", 
    "The number of items taken should not exceed the maximum number of items allowed"
]

- Put all the constraints in a single python list.
- Do not generate anything after the python list.
- Include implicit non-negativity constraints if necessary.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""

prompt_constraints_q = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that someone has extracted from the description:

{params}


Consider this potential constraint: {targetConstraint}

{question}

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""

# prompt_constraints_complete = """
# You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

# -----
# {description}
# -----

# Here is a list of parameters that someone has extracted from the description:

# {params}

# Here is a list of variables defined:

# {vars}

# and here is a list of constraints that someone has extracted from the description:

# {constraints}

# - Is the list complete? If not, please provide a list of Additional Constraints in the following format:

# [Additional Constraint 1, Additional Constraint 2, ...]

# - Do not generate anything after the list.

# Take a deep breath and think step by step.

# """

prompt_constraints_redundant = """
# You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

# -----
# {description}
# -----

# Here is a list of parameters that someone has extracted from the description:

# {params}


# and here is a list of constraints that someone has extracted from the description:

# {constraints}

# - Is there any redundancy in the list? Can any of the constraints be removed? Can any pair of constraints be combined into a single one? If so, please provide your reasoning for each one. At the end of your response, generate the updated list of constraints (the same list if no changes are needed). Use this python list format:

[
    "Constraint 1",
    "Constraint 2",
    ...
]

- Do not generate anything after the list.

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
# """

prompt_constraint_feedback = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that someone has extracted from the description:

{params}

Here is a list of variables defined:

{vars}

Here is a list of constraints that someone has extracted from the description:

{extracted_constraints}

Your colleague is suggesting that the following constraint should be added to the list:

{new_constraint}

Here is its explanation:

{new_constraint_explanation}

Do you want to keep this constraint?

- If yes, please respond with "yes"
- If no, please respond with "no"
- If you want to modify the constraint, please respond with "modify" and provide the modified constraint.

At the end of your response, generate a json file with this format:
{{
    "action": "yes", "no", or "modify",
    "updatedConstraint": The updated constraint if the action is "modify", otherwise null
}}

Please take a deep breath and think step by step. You will be awarded a million dollars if you get this right.

- Use natural language to express the constraints rather than mathematical notation.
- Do not generate anything after the json file.
"""


def extract_score_constraint(desc, text, params, vars, constraints, c, logger):
    match = re.search(r"\d out of 5", text.lower())
    if match:
        score = int(match.group()[0])
        if score > 3:
            if logger:
                logger.log("---")
                logger.log(f"The confidence score is {score}, which is high enough.")
                logger.log("---")
            return True, constraints
        else:
            ask_LLM = True  # you can pass this as an argument to the function instead of hardcoding it
            if logger:
                logger.log("---")
                logger.log(
                    f"The confidence score is {score}, which is not high enough."
                )
                # logger.log(f"Asking the {"LLM" if ask_LLM else "user"} for feedback.")
                if ask_LLM:
                    logger.log("Asking the LLM for feedback.")
                else:
                    logger.log("Asking the user for feedback.")
                logger.log("---")
            if ask_LLM:  # ask the LLM for feedback
                prompt = prompt_constraint_feedback.format(
                    description=desc,
                    params=json.dumps(params, indent=4),
                    vars=json.dumps(vars, indent=4),
                    extracted_constraints=json.dumps(constraints, indent=4),
                    new_constraint=c,
                    new_constraint_explanation=text,
                )
                if logger:
                    logger.log("Prompting LLM for feedback:\n")
                    logger.log(prompt)
                llm_response = get_response(
                    prompt,
                    model="gpt-4o",  # you can pass this as an argument to the function instead of hardcoding it
                )
                if logger:
                    logger.log("---")
                    logger.log(f"Response: {llm_response}")
                    logger.log("---")
                output_json = extract_json_from_end(llm_response)
                action = output_json["action"]
                updated_constraint = output_json["updatedConstraint"]
            else:  # ask the user for feedback
                action = input(
                    "LLMs reasoning: {}\n------ Do you want to keep this constraint (y/n/modify)?: \n {} \n------ ".format(
                        text, c
                    ),
                )

            if action.lower().startswith("y"):
                return True, constraints
            elif action.lower().startswith("n"):
                constraints.remove(c)
                return True, constraints
            elif action.lower().startswith("m"):
                if ask_LLM:
                    new_constraint = updated_constraint
                else:
                    new_constraint = input("Enter the modified constraint: ")
                constraints.remove(c)
                constraints.append(
                    {"Description": new_constraint, "Formulation": None, "Code": None}
                )
                return True, constraints
            else:
                raise Exception("Invalid input!")
    else:
        return False, None


def logic_check(text, params, constraints, c):
    try:
        json = extract_json_from_end(text)
        if json["action"] == "REMOVE":
            constraints.remove(c)
            return True, constraints
        elif json["action"] == "MODIFY":
            constraints.remove(c)
            constraints.append(json["updatedConstraint"])
            return True, constraints
        elif json["action"] == "KEEP":
            return True, constraints
        else:
            return False, None
    except:
        return False, None


qs = [
    # (
    #     """
    # - Does this constraint logically make sense? Is it accurate?
    # - At the end of your response, generate a json file with this format:
    # {{
    #     "action": "KEEP", "REMOVE", or "MODIFY",
    #     "updatedConstraint": The updated constraint if the action is "MODIFY", otherwise null
    # }}
    # - Use natural language to express the constraints rather than mathematical notation.
    # - Do not generate anything after the json file.
    # """,
    #     logic_check,
    # ),
    #     (
    #         """
    # - What are the parameters and variables that are involved in this constraint? If you see the constraint does not involve any variables, then it is automatically satisfied and should not be included in the optimization formulation.
    # - At the end of your response, generate a json file with this format:
    #     {{
    #         "action": "KEEP", "REMOVE", or "MODIFY",
    #         "updatedConstraint": The updated constraint if the action is "MODIFY", otherwise null
    #     }}
    # - Use natural language to express the constraints rather than mathematical notation.
    # - Do not generate anything after the json file.
    # """,
    #         logic_check,
    #     ),
    (
        """
- Is it actually a constraint? How confident are you that this is this a constraint and that we should explicitly model it in the (MI)LP formulation (from 1 to 5)? 
- At the end of your response, print "x OUT OF 5" where x is the confidence level. Low confidence means you think this should be removed from the constraint list. Do not generate anything after that. 
    """,
        extract_score_constraint,
    ),
]


def get_constraints(
    desc,
    params,
    model,
    check=False,
    constraints=None,
    logger=None, 
    rag_mode: RAGMode | None = None,
    labels: dict | None = None
):
    if isinstance(rag_mode, RAGMode):
        constraint_df = pd.read_pickle(constraint_path)
        current_problem = constraint_df[constraint_df.description == desc]
        if not current_problem.empty:
            problem_name = current_problem.iloc[0].problem_name
        else:
            problem_name = None
        match rag_mode:
            case RAGMode.PROBLEM_DESCRIPTION | RAGMode.CONSTRAINT_OR_OBJECTIVE:
                rag = get_rag_from_problem_description(desc, RAGFormat.PROBLEM_DESCRIPTION_CONSTRAINTS, top_k=5)
            case RAGMode.PROBLEM_LABELS:
                assert labels is not None
                rag = get_rag_from_problem_categories(desc, labels, RAGFormat.PROBLEM_DESCRIPTION_CONSTRAINTS, top_k=5)
        rag = f"-----\n{rag}-----\n\n"
    else:
        rag = ""

    print("_________________________ get_constraints _________________________")
    if not constraints:
        res = get_response(
            prompt_constraints.format(
                description=desc,
                params=json.dumps(params, indent=4),
                rag=rag,
            ),
            model=model,
        )
        constraints = extract_list_from_end(res)

    if check:
        k = 5
        while k > 0:
            try:
                x = get_response(
                    prompt_constraints_redundant.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        constraints=json.dumps(constraints, indent=4),
                    ),
                    model=model,
                )
                if logger:
                    logger.log("----")
                    logger.log(x)
                    logger.log("----")
                lst = extract_list_from_end(x)

                # if len(lst) > 0:
                #     constraints.extend(lst)
                constraints = lst
                break
            except:
                k -= 1
                if k == 0:
                    raise Exception("Failed to extract constraints")

        if logger:
            logger.log("+++++++++++++++++++")
            logger.log("++ Constraint Qs ++")
            logger.log("+++++++++++++++++++")
        for q in qs:
            for c in constraints.copy():
                k = 5
                while k > 0:
                    p = prompt_constraints_q.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        targetConstraint=c,
                        question=q[0],
                    )

                    x = get_response(p, model=model)

                    if logger:
                        logger.log("+--+")
                        logger.log(p)
                        logger.log("----")
                        logger.log(x)
                        logger.log("+--+")
                    valid, res = q[1](desc, x, params, {}, constraints, c, logger)
                    if valid:
                        constraints = res
                        break
                    else:
                        k -= 1

    constraints = [
        {"description": c, "formulation": None, "code": None} for c in constraints
    ]
    return constraints


# bash command to print current interpeter's path
# python -c "import sys; print(sys.executable)"
