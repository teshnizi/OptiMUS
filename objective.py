import json
import re
from rag.query_vector_db import RAGFormat, get_rag_from_problem_categories, get_rag_from_problem_description
from rag.rag_utils import RAGMode
from utils import (
    extract_list_from_end,
    get_response,
    extract_json_from_end,
    extract_equal_sign_closed,
)


def extract_objective(text):

    # find first and second occurence of "=====" in the text
    ind_1 = text.find("=====")
    ind_2 = text.find("=====", ind_1 + 1)

    # extract the text between the two "=====" occurences
    objective = text[ind_1:ind_2]
    objective = objective.replace("=====", "").strip()
    objective = objective.replace("OBJECTIVE:", "").strip()
    return objective


prompt_objective = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

{rag}-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}

Your task is to identify and extract the optimization objective from the description. The objective is the goal that the optimization model is trying to achieve (e.g. maximize profit, minimize cost). Please generate the output in the following format:

=====
OBJECTIVE: objective description
=====

for example:

=====    
OBJECTIVE: "The goal is to maximize the total profit from producing television sets"
=====

- Do not generate anything after the objective.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def get_objective(
    desc,
    params,
    model,
    check=False,
    logger=None,
    rag_mode: RAGMode | None = None,
    labels: dict | None = None,
):
    if isinstance(rag_mode, RAGMode):
        match rag_mode:
            case RAGMode.PROBLEM_DESCRIPTION | RAGMode.CONSTRAINT_OR_OBJECTIVE:
                rag = get_rag_from_problem_description(desc, RAGFormat.PROBLEM_DESCRIPTION_OBJECTIVE, top_k=5)
            case RAGMode.PROBLEM_LABELS:
                assert labels is not None
                rag = get_rag_from_problem_categories(desc, labels, RAGFormat.PROBLEM_DESCRIPTION_OBJECTIVE, top_k=5)
        rag = f"-----\n{rag}-----\n\n"
    else:
        rag = ""
    res = get_response(
        prompt_objective.format(
            description=desc,
            params=json.dumps(params, indent=4),
            rag=rag,
        ),
        model=model,
    )
    objective = extract_objective(res)

    objective = {"description": objective, "formulation": None, "code": None}

    return objective
