"""
Implement different utilities with OptiMUS v0.3
"""

import os
import time
import json
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from typing import Dict

from parameters import get_params
from constraint import get_constraints
from constraint_model import get_constraint_formulations
from target_code import get_codes
from generate_code import generate_code
from utils import load_state, save_state, Logger
from objective import get_objective
from objective_model import get_objective_formulation
from execute_code import execute_and_debug
from utils import create_state, get_labels

ERROR_CORRECTION = False
MODEL = "gpt-4o"
RAG_MODE = None
DEFAULT_LABELS = {"types": ["Mathematical Optimization"], "domains": ["Operations Management"]}

def get_intro_latex_code_map(fname) -> Dict:
    """
    Extract the internal OptiMUS data structure that maps natural language, LaTeX and code
    No data is required by this utility
    
    """
    
    run_dir = "."
    
    # Extract parameters from the natural language description
    with open(fname, "r") as f:
        desc = f.read()
        f.close()
        
    params = get_params(desc, check=True)
    
    # Read the description
    with open(fname, "r") as f:
        desc = f.read()
    
    # Construct the initial state
    state = {"description": desc, "parameters": params}
    
    save_state(state, "state_1_params.json")

    logger = Logger(f"log.txt")
    logger.reset()
    logger = Logger(f"log.txt")
    logger.reset()
    
    # Get objective
    state = load_state(os.path.join(run_dir, "state_1_params.json"))
    objective = get_objective(
        state["description"],
        state["parameters"],
        check=ERROR_CORRECTION,
        logger=logger,
        model=MODEL,
        rag_mode=RAG_MODE,
        labels=DEFAULT_LABELS,
    )
    print(objective)
    state["objective"] = objective
    save_state(state, os.path.join(run_dir, "state_2_objective.json"))
    
    # Get constraints
    state = load_state(os.path.join(run_dir, "state_2_objective.json"))
    constraints = get_constraints(
    state["description"],
    state["parameters"],
    check=ERROR_CORRECTION,
    logger=logger,
    model=MODEL,
    rag_mode=RAG_MODE,
    labels=DEFAULT_LABELS,
    )
    print(constraints)
    state["constraints"] = constraints
    save_state(state, os.path.join(run_dir, "state_3_constraints.json"))
    
    # Get constraint formulations
    state = load_state(os.path.join(run_dir, "state_3_constraints.json"))
    constraints, variables = get_constraint_formulations(
        state["description"],
        state["parameters"],
        state["constraints"],
        check=ERROR_CORRECTION,
        logger=logger,
        model=MODEL,
        rag_mode=RAG_MODE,
        labels=DEFAULT_LABELS,
    )
    state["constraints"] = constraints
    state["variables"] = variables
    save_state(state, os.path.join(run_dir, "state_4_constraints_modeled.json"))
    
    # Get objective formulation
    state = load_state(os.path.join(run_dir, "state_4_constraints_modeled.json"))
    objective = get_objective_formulation(
        state["description"],
        state["parameters"],
        state["variables"],
        state["objective"],
        model=MODEL,
        check=ERROR_CORRECTION,
        rag_mode=RAG_MODE,
        labels=DEFAULT_LABELS,
    )
    state["objective"] = objective
    print("DONE OBJECTIVE FORMULATION")
    save_state(state, os.path.join(run_dir, "state_5_objective_modeled.json"))

    # # ####### Get codes
    state = load_state(os.path.join(run_dir, "state_5_objective_modeled.json"))
    constraints, objective = get_codes(
        state["description"],
        state["parameters"],
        state["variables"],
        state["constraints"],
        state["objective"],
        model=MODEL,
        check=ERROR_CORRECTION,
    )
    state["constraints"] = constraints
    state["objective"] = objective
    save_state(state, os.path.join(run_dir, "state_6_code.json"))
    
    return state

if __name__ == "__main__":
    
    get_intro_latex_code_map("./description.txt")