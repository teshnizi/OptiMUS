"""
Define templates for json file
"""
import json
from typing import List, Dict, Union
import openai
import numpy as np
from mistralai.client import MistralClient


INPUT_FIELD_DEF = "definition"
INPUT_FIELD_SYMBOL = "symbol"
INPUT_FIELD_FORMULATION = "formulation"
INPUT_FIELD_CODE = "code"
INPUT_FIELD_INDEX = "index"
INPUT_FIELD_DIMENSION = "shape"
INPUT_FIELD_CONSTR_VAR = "related_variables"
INPUT_FIELD_CONSTR_PARAMS = "related_parameters"
INPUT_FIELD_CONSTR_STATUS = "status"

NLP_PARAM_CONSTANT = "constant"

NLP_PROBLEM_LP = "linear programming"
NLP_PROBLEM_MILP = "mixed integer linear programming"
NLP_PROBLEM_UNKNOWN = "unknown"


class NLParamParser(object):
    """
    Modeling interface for Optimus v2.
    The interface collects the user input as natural language parameter builder

    """

    def __init__(self, problem_type: str = NLP_PROBLEM_UNKNOWN) -> None:
        """

        :param problem_type: Type of the optimization problem
        """

        # Data parameters for modeling
        self._dimensions = {}  # type: Dict[str: Dict]
        self._params = {}  # type: Dict[str: Dict]

        self._background = ""  # type: str

        self._n_constrs = 0
        self._constraints = {}  # type: Dict[str: Dict]
        self._objective = {}  # type: Dict

        self._problem_type = problem_type  # type: str

        return

    def add_background(self, nlp_def: str):
        self._background = nlp_def

    def _create_dimension(self, nlp_def: str, symbol: str) -> str:
        optimus_nlp_dim = "Optimus_dim_{0}".format(symbol)
        self._dimensions[optimus_nlp_dim] = {
            INPUT_FIELD_DEF: nlp_def,
            INPUT_FIELD_SYMBOL: symbol,
        }
        return optimus_nlp_dim

    def add_param(self, nlp_def: str, symbol: str, param_dim: List) -> None:
        """
        Add a parameter to the parameter pool
        :param nlp_def: The NLP definition of a piece of data parameter
        :param symbol: The symbol for the parameter
        :param param_dim: Dimension of the data. Could be NLP_PARAM_CONSTANT or NLP_PARAM_DIMENSION
        :return:
        """

        # if param_dim != NLP_PARAM_CONSTANT:
        #     for d in param_dim:
        #         if d not in self._dimensions.keys():
        #             raise RuntimeError("Dimension %d is not existent" % d)

        optimus_nlp_param = "Optimus_param_{0}".format(symbol)

        if optimus_nlp_param in list(self._params.keys()):
            raise KeyError("Parameter %s already defined" % symbol)

        self._params[optimus_nlp_param] = {
            INPUT_FIELD_DEF: nlp_def,
            INPUT_FIELD_SYMBOL: symbol,
            INPUT_FIELD_DIMENSION: param_dim,
        }

        return

    def add_constraint(self, nlp_def: str) -> None:
        optimus_nlp_constr = "Optimus_constr_{0}".format(self._n_constrs)
        self._n_constrs += 1

        self._constraints[optimus_nlp_constr] = {
            INPUT_FIELD_DEF: nlp_def,
            INPUT_FIELD_FORMULATION: "",
            INPUT_FIELD_CODE: "",
            INPUT_FIELD_INDEX: self._n_constrs,
            INPUT_FIELD_CONSTR_VAR: [],
            INPUT_FIELD_CONSTR_PARAMS: [],
            INPUT_FIELD_CONSTR_STATUS: "not_formulated",
        }

        return

    def set_objective(self, nlp_def: str) -> None:
        self._objective = {
            INPUT_FIELD_DEF: nlp_def,
            INPUT_FIELD_FORMULATION: "",
            INPUT_FIELD_CODE: "",
            INPUT_FIELD_INDEX: self._n_constrs,
            INPUT_FIELD_CONSTR_VAR: [],
            INPUT_FIELD_CONSTR_PARAMS: [],
            INPUT_FIELD_CONSTR_STATUS: "not_formulated",
        }

        return

    def get_summary(self) -> None:
        print("Problem background:")
        print(self._background)

        print("Dimension info:")
        for k, v in self._dimensions.items():
            print("%40s: %s" % (v[INPUT_FIELD_DEF], v[INPUT_FIELD_SYMBOL]))

        print("Parameters:")
        for k, v in self._params.items():
            print("%40s: %s" % (v[INPUT_FIELD_DEF], v[INPUT_FIELD_SYMBOL]))

        print("Constraints:")
        for k, v in self._constraints.items():
            print("%40s: %s" % (v[INPUT_FIELD_DEF], v[INPUT_FIELD_FORMULATION]))

        print("Objective:")
        print("%40s" % self._objective[INPUT_FIELD_DEF])

    @staticmethod
    def prep_problem_json(state):
        for parameter in state["parameters"]:
            assert "shape" in parameter.keys(), "shape is not defined for parameter"
            assert "symbol" in parameter.keys(), "symbol is not defined for parameter"
            assert (
                "definition" in parameter.keys() and len(parameter["definition"]) > 0
            ), "definition is not defined for parameter"

            if parameter["shape"]:
                code_symbol = parameter["symbol"].split("_")[0]
                parameter[
                    "code"
                ] = f'{code_symbol} = np.array(data["{code_symbol}"]) # {parameter["shape"]}'
            else:
                code_symbol = parameter["symbol"].split("_")[0]
                parameter[
                    "code"
                ] = f'{code_symbol} = data["{code_symbol}"] # scalar parameter'

        return state

    def get_initial_state(self) -> Dict:
        """
        Prepare the input format for Optimus agents
        :return:
        """

        state = {
            "background": self._background,
            "problem_type": self._problem_type,
            # "dimensions": list(self._dimensions.values()),
            "parameters": list(self._params.values()),
            "constraints": list(self._constraints.values()),
            "variables": [],
            "objective": [self._objective],
            "solution_status": None,
            "solver_output_status": None,
            "error_message": None,
            "obj_val": None,
            "data_json_path": "data.json",
        }

        return self.prep_problem_json(state)


def sanity_check(state):
    # read the data from file:

    assert (
        "data_json_path" in state.keys()
    ), "data_json_path is not defined in the state"

    with open(state["data_json_path"], "r") as f:
        data = json.load(f)

    for param in state["parameters"]:
        if not "shape" in param.keys():
            raise KeyError(f"shape is not defined for parameter {param['definition']}")

        if not "symbol" in param.keys():
            raise KeyError(f"symbol is not defined for parameter {param['definition']}")

        if len(param["symbol"].split("_")) > 2:
            raise KeyError(
                f"Please use camelCase for parameter symbols! Error in {param['symbol']}"
            )

        if not "definition" in param.keys():
            raise KeyError(
                f"definition is not defined for parameter {param['definition']}"
            )

        symb = param["symbol"].split("_")[0]
        if not symb in data.keys():
            raise KeyError(f"{param['symbol']} is not defined in data.json")

        pd = np.array(data[symb])
        if param["shape"] != []:
            for idx, dim in enumerate(param["shape"]):
                if not dim in data:
                    raise KeyError(
                        f"{dim} is not defined in data.json, but is used in {param['symbol']}"
                    )

                if data[dim] != pd.shape[idx]:
                    raise ValueError(
                        f"Dimension mismatch for {param['symbol']} at dim {idx}"
                    )

    #    make sure that parameter codes are defined
    for param in state["parameters"]:
        if not "code" in param.keys():
            raise KeyError(f"code is not defined for parameter {param['definition']}")


def get_openai_client():
    with open("config.json") as f:
        config = json.load(f)
    if len(config["openai_api_key"]) < 10:
        raise ValueError("Please provide a valid OpenAI API key in config.json")
    config["openai_api_key"]

    client = openai.Client(
        api_key=config["openai_api_key"], organization=config["openai_org"]
    )

    return client


def get_tai_client():
    with open("config.json") as f:
        config = json.load(f)
    if len(config["together_api_key"]) < 10:
        raise ValueError("Please provide a valid Together API key in config.json")

    client = openai.OpenAI(
        api_key=config["together_api_key"],
        base_url="https://api.together.xyz",
    )

    return client


def get_mistral_client():
    with open("config.json") as f:
        config = json.load(f)
    if len(config["mistral_api_key"]) < 10:
        raise ValueError("Please provide a valid Mistral API key in config.json")

    client = MistralClient(api_key=config["mistral_api_key"])

    return client


if __name__ == "__main__":
    parser = NLParamParser()
