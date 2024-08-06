import json
from utils import get_response, extract_json_from_end, shape_string_to_list
import pandas as pd
from rag.query_vector_db import RAGFormat, get_rag_from_constraint, get_rag_from_problem_categories, get_rag_from_problem_description
from rag.rag_utils import RAGMode, constraint_path


def extract_formulation_from_end(text):

    ind_1 = text.find('"FORMULATION":')

    iloop = 0
    iloop_max = 1e05

    if "$" not in text:
        raise Exception("No formulation found")

    while text[ind_1] != "$":
        ind_1 += 1
        iloop += 1
        if iloop > iloop_max:
            raise Exception("No formulation found")

    ind_2 = text.find("NEW VARIABLES")
    while text[ind_2] != "$":
        ind_2 -= 1
        iloop += 1
        if iloop > iloop_max:
            raise Exception("No formulation found")

    formulation = text[ind_1 : ind_2 + 1].strip()

    text = text[:ind_1] + text[ind_2 + 1 :]

    ind_1 = text.find('"AUXILIARY CONSTRAINTS":')
    while text[ind_1] != "$":
        ind_1 += 1
        if ind_1 > len(text) - 1:
            break
        iloop += 1
        if iloop > iloop_max:
            raise Exception("No formulation found")

    auxiliaries = []

    if ind_1 < len(text) - 1:
        while True:

            ind_2 = ind_1 + 1

            while ind_2 + 2 < len(text) and text[ind_2 : ind_2 + 2] != '$"':
                ind_2 += 1
                iloop += 1
                if iloop > iloop_max:
                    break
                    raise Exception("No formulation found")
            auxiliaries.append(text[ind_1 : ind_2 + 1].strip())
            text = text[:ind_1] + text[ind_2 + 1 :]

            while ind_1 < len(text) - 1 and text[ind_1] != "$":
                ind_1 += 1
                if ind_1 > len(text) - 1:
                    break
                iloop += 1
                if iloop > iloop_max:
                    break
                    raise Exception("No formulation found")

            if ind_1 > len(text) - 1:
                break
    json_res = extract_json_from_end(text)

    auxiliaries = [a for a in auxiliaries if len(a) > 5]
    json_res["FORMULATION"] = formulation
    json_res["AUXILIARY CONSTRAINTS"] = auxiliaries

    return (
        json_res["FORMULATION"],
        json_res["NEW VARIABLES"],
        json_res["AUXILIARY CONSTRAINTS"],
    )


text = """
To model the optimization constraint "The amount of stock held from one period to the next incurs a holding cost," we need to first understand that the holding cost is a cost incurred for maintaining stock in the warehouse from one period to the next. This cost is proportional to the amount of stock held and the holding cost per unit.

Let’s denote:
- $\text{holdingCost}$ as the cost of holding one unit of product per period.
- $\text{stockLevel}_n$ as the stock level at the end of period $n$.

The holding cost for one period is then given by $\text{holdingCost} \times \text{stockLevel}_n$. Our goal is to include this cost in the objective function of our MILP formulation, affecting the profit maximization.

However, since the request is for a constraint formulation rather than directly incorporating this cost into the objective function, the likely intent is to ensure that the stock levels and their associated costs are properly tracked. While directly constraining the holding cost is unconventional without specific cost limits, we ensure proper stock level management through auxiliary constraints.

Here’s the step-by-step approach to formulation:

1. **Track stock levels**: Ensure the stock level in each period is updated correctly based on purchases and sales.

Now, I'll generate the output JSON file with the required constraint formulation:

```json
{
    "FORMULATION": "$\forall n, \\text{holdingCost} \times \\text{stockLevel}[n]$",
    "NEW VARIABLES": {},
    "AUXILIARY CONSTRAINTS": [
        "$\\text{profit} = \sum_{n=1}^{NumPeriods} \\left(\\text{Price}[n] \times \\text{sellAmount}[n] - \\text{Cost}[n] \times \\text{buyAmount}[n] - \\text{holdingCost} \times \\text{stockLevel}[n]\\right)$",
        "$\\text{profit} = \\text{maximize}(\\text{profit})$"
    ]
}
```

Explanation:
1. **FORMULATION**: This is the mathematical representation showing that holding cost is calculated as $\text{holdingCost} \times \text{stockLevel}[n]$.

2. **NEW VARIABLES**: No new variables are introduced.

3. **AUXILIARY CONSTRAINTS**:
   - The first auxiliary constraint shows the objective function incorporating the holding cost into the profit calculation.
   - The second auxiliary constraint states the maximization of profit.

Note: The initial constraint provided seems to be more of a statement for incorporating holding costs into the profit function rather than a strict boundary constraint, and thus is reflected appropriately in the auxiliary constraints.
"""

prompt_constraints_model = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{rag}-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}

And here's a list of all variables that we have defined so far to model the problem as an (MI)LP:

{vars}

Your task is to model the following constraint mathematically in LaTeX for the MILP formulation:

{constraint}

The constraints are the conditions that must be satisfied by the variables. Please generate the output in the following json format:

{{
    "FORMULATION": constraint formulation in LaTeX, between $...$,
    "NEW VARIABLES": {{
        symbol: {{    
            "shape": shape of the new variable (e.g. [], [N], [N, M]),
            "type": type of the new variable (e.g. binary, integer, continuous),
            "definition": definition of the new variable in natural language
        }},
        ...
    }},
    "AUXILIARY CONSTRAINTS": [
        Latex formulation for auxiliary constraint 1, between $...$,
        Latex formulation for auxiliary constraint 2, between $...$,
        ...
    ]
}}
    
Here's an example output (where SalesVolumePerStore is already defined as a variable in the vars list):
{{
    "FORMULATION": "$\\forall i, SalesVolumes[i] \leq MaxProductionVolumes[i]$",
    "NEW VARIABLES": {{
        "SalesVolumes": {{
            "shape": "[NumberOfArticles]",
            "type": "continuous",
            "definition": "The sales volume for each article of clothing"
        }}
    }},
    "AUXILIARY CONSTRAINTS": [
        "$\\forall i, SalesVolumes[i] = \\sum_j SalesVolumesPerStore[i, j]$"
    ]
}}

- If you need any new variables, you can define them in the NEW VARIABLES list. Use {{}} for "NEW VARIABLES" if no new variables are needed.
- Use [] for AUXILIARY CONSTRAINTS list if no auxiliary constraints are needed.
- You can only use symbols of existing parameters and integer numbers for dimensions of new variables.
- Use camelCase for variable symbols (e.g. SalesVolumes). Do not use LaTeX formatting (e.g. X_{{color}}), indices (e.g. SalesVolume_{{i}}), and underlines (_) for variable symbols.
- Do not generate anything after the json file!

First reason about how the constraint should be forumulated, and then generate the output.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


prompt_constraints_q = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that someone has extracted from the description:

{params}

And here is a list of variables defined:

{vars}

Consider this constraint:

{targetConstraint}

{question}

 Take a deep breath and think step by step.
"""


prompt_constraints_aux = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

Here is a list of parameters that someone has extracted from the description:

{params}

And here is a list of variables  defined:

{vars}

Consider this constraint: {targetConstraint}

To define this constraint, we have defined these new variables:

{new_vars}

- Are there any auxiliary constraints we need to define on these variables to make sure they are consistent with the other variables and constraints (on top of the main constraint)? Generate a list of these constraints in this format:

[
    Latex formulation for auxiliary constraint 1,
    Latex formulation for auxiliary constraint 2,
    ...
]


- First reason about what auxiliary constraints we need, and then generate the list. Do not generate anything after the list. 

Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def logic_check(text, params, vars, constraints, c):
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


def extract_score_constraint_model(text, params, vars, constraints, c):
    match = re.search(r"\d out of 5", text.lower())
    if match:
        score = int(match.group()[0])
        if score > 3:
            return True, constraints
        else:
            inp = input(
                "LLMs reasoning: {}\n------ Do you want to keep this constraint (y/n/modify)?: \n {} \n------ ".format(
                    text, c
                ),
            )
            if inp.lower().startswith("y"):
                return True, constraints
            elif inp.lower().startswith("n"):
                constraints.remove(c)
                return True, constraints
            elif inp.lower().startswith("m"):
                new_constraint = input("Enter the modified formulation: ")
                constraints.remove(c)
                constraints.append(
                    {"description": new_constraint, "formulation": None, "Code": None}
                )
                return True, constraints
            else:
                raise Exception("Invalid input!")
    else:
        return False, None


qs = [
    (
        """
    - Does this constraint logically make sense? How confident are you that this needs to be explicitly modeled in the optimization formulation (from 1 to 5)? 
    - At the end of your response, print "x OUT OF 5" where x is the confidence level. Do not generate anything after that. 
    """,
        # extract_score,
        # dummy function
        lambda x, params, vars, constraints, c: (False, constraints),
    ),
    (
        """
- What are the units for each side of the constraint? Are they consistent with each other?
- At the end of your response, generate a json file with this format:
    {{
        "action": "KEEP" if the units match, or "MODIFY" if the units do not match,
        "updatedConstraint": The latex code for updated constraint if the action is "MODIFY", otherwise null
    }}

- Do not generate anything after the json file.

""",
        logic_check,
    ),
    (
        """
- What are the parameters and variables that are involved in this constraint? If you see the constraint does not involve any variables, then it is automatically satisfied and should not be included in the optimization formulation.
- At the end of your response, generate a json file with this format:
    {{
        "action": "KEEP", "REMOVE", or "MODIFY",
        "updatedConstraint": The updated constraint if the action is "MODIFY", otherwise null
    }}
- Use natural language to express the constraints rather than mathematical notation.
- Do not generate anything after the json file.
""",
        logic_check,
    ),
]


def get_constraint_formulations(
    desc,
    params,
    constraints,
    model,
    check=False,
    logger=None,
    rag_mode: RAGMode | None = None,
    labels: dict | None = None
):
    if isinstance(rag_mode, RAGMode):
        match rag_mode:
            case RAGMode.PROBLEM_DESCRIPTION:
                rag = get_rag_from_problem_description(desc, RAGFormat.CONSTRAINT_FORMULATION, top_k=5)
            case RAGMode.CONSTRAINT_OR_OBJECTIVE:
                rag = ""
            case RAGMode.PROBLEM_LABELS:
                assert labels is not None
                rag = get_rag_from_problem_categories(desc, labels, RAGFormat.CONSTRAINT_FORMULATION, top_k=5)
        rag = f"-----\n{rag}-----\n\n"
    else:
        rag = ""

    if logger:
        logger.log("\n\n\n++++++++++++++++++++++++++++++")
        logger.log("Extracting constraint formulations")
        logger.log("++++++++++++++++++++++++++++++\n\n\n")

    vars = {}
    formulated_constraints = []
    for c in constraints.copy():
        k = 1
        while k > 0:
            try:
                if rag_mode == RAGMode.CONSTRAINT_OR_OBJECTIVE:
                    constraint_df = pd.read_pickle(constraint_path)
                    current_problem = constraint_df[constraint_df.description == desc]
                    if not current_problem.empty:
                        problem_name = current_problem.iloc[0].problem_name
                    else:
                        problem_name = None
                    rag = get_rag_from_constraint(c["description"], RAGFormat.CONSTRAINT_FORMULATION, top_k=10, current_problem_name=problem_name)
                    rag = f"-----\n{rag}-----\n\n"
                res = get_response(
                    prompt_constraints_model.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        vars=json.dumps(vars, indent=4),
                        constraint=c,
                        rag=rag,
                    ),
                    model=model,
                )

                if logger:
                    logger.log("----")
                    logger.log(res)
                    logger.log("----")

                formulation, new_variables, aux_constraints = (
                    extract_formulation_from_end(res)
                )

                if logger:
                    logger.log("----")
                    logger.log("EXTRACTED ITEMS")
                    logger.log(str(formulation))
                    logger.log(str(new_variables))
                    logger.log(str(aux_constraints))
                    logger.log("----")

                tmp_vars = vars.copy()
                for v in new_variables:
                    if v in tmp_vars:
                        raise Exception(f"Variable {v} already exists")
                    print(v, new_variables[v])
                    new_variables[v]["shape"] = shape_string_to_list(
                        new_variables[v]["shape"]
                    )
                    tmp_vars[v] = new_variables[v]

                c["formulation"] = formulation
                formulated_constraints.append(c)

                for aux_c in aux_constraints:
                    formulated_constraints.append(
                        {"description": "auxiliary constraint", "formulation": aux_c}
                    )

                vars = tmp_vars
                break
            except Exception as e:
                k -= 1
                if k == 0:
                    raise (e)

    constraints = formulated_constraints

    if check:
        for c in formulated_constraints.copy():
            for q in qs[0:1]:
                k = 1
                while k > 0:
                    p = prompt_constraints_q.format(
                        description=desc,
                        params=json.dumps(params, indent=4),
                        vars=json.dumps(vars, indent=4),
                        targetConstraint=json.dumps(c, indent=4),
                        question=q[0],
                    )

                    x = get_response(p, model=model)

                    valid, res = q[1](x, params, vars, constraints, c)

                    print(valid)
                    if valid:
                        constraints = res
                        break
                    else:
                        k -= 1

    return formulated_constraints, vars
