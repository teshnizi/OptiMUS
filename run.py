import argparse
import time
import os
import re

from utils.misc import *

from agents.agent import Agent
from agents.manager import GroupChatManager
from agents.user_proxy import UserProxy
from agents.formulator import Formulator
from agents.programmer import Programmer
from agents.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Run the algorithm on the dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        help='Dataset name, "nl4opt" or "ComplexOR" or "nlp4lp"',
        default="nl4opt",
    )
    parser.add_argument("--problem", type=str, help="Problem name", default="prob_1")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-1106-preview",
        help="Base large language model",
    )
    parser.add_argument("--log_dir", type=str, help="Log directory")

    parser.add_argument(
        "--max_selections",
        type=int,
        default=8,
        help="Number of max agent selections",
    )
    args = parser.parse_args()

    if not args.model in [
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral-medium",
    ]:
        print(
            "Invalid model name! Please choose from 'gpt-4-1106-preview', 'gpt-3.5-turbo', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistral-medium'"
        )
        exit(0)

    dataset = args.dataset.lower()
    problem = args.problem

    dir = os.path.join("data", args.dataset)

    print(dir)
    if not os.path.exists(dir):
        print(f"Dataset {args.dataset} not found!")
        exit(0)
    if not os.path.exists(os.path.join(dir, problem)):
        print(f"Problem {problem} not found!")
        exit(0)

    # create the log directory
    log_dir = args.log_dir
    print(log_dir)
    if log_dir is None:
        log_dir = f"logs/log_{time.strftime('%Y%m%d%H%M%S')}_{dataset}_{problem}/"
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    openai_client = get_openai_client()
    tai_client = get_tai_client()
    mistral_client = get_mistral_client()

    openai_client = get_openai_client()
    tai_client = get_tai_client()
    mistral_client = get_mistral_client()

    if args.model.startswith("gpt"):
        client = openai_client
    elif args.model.startswith("mistral-medium"):
        client = mistral_client
    else:
        client = tai_client

    formulator = Formulator(
        client=client,
        llm=args.model,
    )

    programmer = Programmer(client=client, llm=args.model)

    evaluator = Evaluator(client=client, llm=args.model)

    manager = GroupChatManager(
        client=client,
        agents=[
            formulator,
            programmer,
            evaluator,
            #    user_proxy
        ],
        llm=args.model,
    )

    print(f"Solving {args.dataset} problem {problem}...")

    with open(f"data/{args.dataset}/{problem}/input_targets.json", "r") as f:
        state = json.load(f)

    state["objective"] = [state["objective"]]
    for target in ["constraints", "objective"]:
        state[target] = [
            {
                "description": x,
                "status": "not_formulated",
            }
            for x in state[target]
        ]

    state = NLParamParser.prep_problem_json(state)

    state = {
        "background": state["background"],
        "problem_type": "LP",
        "parameters": state["parameters"],
        "constraint": state["constraints"],
        "variables": [],
        "objective": state["objective"],
        "solution_status": None,
        "solver_output_status": None,
        "error_message": None,
        "obj_val": None,
        "log_folder": log_dir,
        "data_json_path": f"data/{dataset}/{problem}/data.json",
    }
    if not os.path.exists(state["log_folder"]):
        os.makedirs(state["log_folder"])

    sanity_check(state)
    try:
        final_state = manager.solve(state=state)
        with open(f"data/{dataset}/{problem}/output.json", "w") as f:
            json.dump(final_state, f, indent=4)

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        with open(f"{state['log_folder']}/error.log", "w") as f:
            f.write(traceback.format_exc())

    print("DONE!")


if __name__ == "__main__":
    main()
