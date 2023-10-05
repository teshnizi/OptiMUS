"""
Implement the interface of applying GPT-based model to solve OR models
"""

import os
import subprocess
import importlib
import argparse

from configure import internal_prompt
from configure import (
    template_formulation,
    template_codegen,
    template_codefix_execution,
    template_codefix_data,
    template_rephrase,
    template_testgen,
    template_standard_prompt,
)

from configure import api_keys

# Import behavior parameters
from configure import (
    # MODE_STANDARD_PROMPT,
    MODE_COT_ONLY,
    MODE_COT_DEBUG,
    MODE_COT_DEBUG_TEST,
    MODE_COT_HUMAN,
)

# Import status code
from configure import (
    STATUS_PASSED,
    STATUS_SYNTAX_ERROR,
    STATUS_LOGIC_ERROR,
)

from typing import List, Dict

from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage
from langchain.chat_models import ChatOpenAI

from utils import (
    read_problem_from_entire_file,
    get_initial_test_script,
    generate_instance_template,
    get_solver_instruction,
)


class GPT4OR(object):
    def __init__(
        self, model_name: str, solver: str, api_key: str, verbose: bool = False
    ) -> None:
        """
        :param model_name: The name of LLM,
        """

        # Which LLM to use
        self._model_name = model_name  # type: str

        # The large language model instance
        self._llm = None  # type: ChatOpenAI
        self._apikey = api_key  # type: str

        # The overall OR model data, including input, output format,
        self._data = {}  # type: Dict[str, str]

        # The path of user specified problem input path/to/description.txt
        self._problem_path = ""  # type: str

        # "You are an expert"
        self._internal_prompt = SystemMessage(content=internal_prompt)

        # The latest LLM response
        self._llm_response = ""  # type: str

        # Response to the formulation request
        self._formulation_response = ""  # type: str

        # The iterations of solver codes
        self._solver_codes = []  # type: List[str]

        # Log of the whole conversation
        self._global_conversations = []

        # The working directory/problem of the agent. The same path as description.txt
        self._path = "."  # type: str
        self._prob_name = ""  # type: str
        self._std_format = "description.txt"  # type: str
        self._std_log = "description.log"  # type: str
        self._std_test = "test-description.py"  # type: str
        self._std_human = "test-human.py"  # type: str

        # Which optimization solver to use?
        self._solver = solver  # type: str

        # The most recent error message
        self._errmsg = ""  # type: str

        # Number of run-and-fix iterations
        self._iter = -1  # type: int

        # Switch on/off each technique
        self._params = {
            "COT": False,  # Whether to use chain of thought
            "Debug": False,  # Whether to apply debugging
            "Test": False,  # Whether to get automatic testing scripts from LLMs
            "Human": False,  # Whether to invoke human testing script
        }  # type: Dict[str, bool]

        self.verbose = verbose

        return

    @staticmethod
    def _log(info):
        # print(info)
        pass

    def _user_says(self) -> None:
        """
        Print header for user input in the log
        :return:
        """

        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("User says: ")
        self._global_conversations.append("------------------------\n")

        return

    def _chatbot_says(self) -> None:
        """
        Print header for LLM input in the log
        :return:
        """

        self._global_conversations.append("\n------------------------")
        self._global_conversations.append("%s says: " % self._model_name)
        self._global_conversations.append("------------------------\n")

        return

    def _model_format(self, template: str) -> str:
        """
        Format the model-side information with data
        :param template: the template to be formatted
        :return: the formatted information
        """

        return template.format(
            PROBLEM_TYPE=self._data["problem_type"],
            PROBLEM_INFO=self._data["problem_info"],
            INPUT_FORMAT=self._data["input_format"],
            OBJECTIVE=self._data["objective_info"],
            OUTPUT_INFO=self._data["output_info"],
            OUTPUT_FORMAT=self._data["output_format"],
            INITIAL_TEST_SCRIPT=self._data["initial_test_script"],
            CODE=self._data["code"],
            SOLVER=self._solver,
            SOLVER_INSTRUCTION=get_solver_instruction(self._solver),
            ERROR_MESSAGE=self._errmsg,
        )

    def _prompt_format(self, template):
        """
        Format the LLM-side information with data
        :param template: the template to be formatted
        :return: the formatted prompt
        """

        return ChatPromptTemplate.from_messages(messages=template).format_messages(
            PROBLEM_TYPE=self._data["problem_type"],
            PROBLEM_INFO=self._data["problem_info"],
            INPUT_FORMAT=self._data["input_format"],
            OBJECTIVE=self._data["objective_info"],
            OUTPUT_INFO=self._data["output_info"],
            OUTPUT_FORMAT=self._data["output_format"],
            INITIAL_TEST_SCRIPT=self._data["initial_test_script"],
            CODE=self._data["code"],
            SOLVER=self._solver,
            SOLVER_INSTRUCTION=get_solver_instruction(self._solver),
            ERROR_MESSAGE=self._errmsg,
        )

    def _connect_chatbot(self) -> None:
        """
        Connect to chat bot
        :return:
        """

        self._llm = ChatOpenAI(
            model_name=self._model_name, temperature=0.3, openai_api_key=self._apikey
        )
        return

    def _get_problem_data(self) -> None:
        """
        Get problem data from description
        :return: The dictionary with problem and in/out information
        """

        self._data = read_problem_from_entire_file(
            os.path.join(self._problem_path, self._std_format)
        )

        # If solving an MILP, we switch to Gurobi
        if (self._data["problem_type"] == "MILP") or (
            self._data["problem_type"] == "ILP"
        ):
            self._solver = "gurobi"

        initial_test_script = get_initial_test_script(
            output_format=self._data["output_format"]
        )

        self._data["initial_test_script"] = initial_test_script

    def _generate_formulation(self) -> None:
        """
        Let LLM generate the mathematical formulation of the instance
        :return:
        """

        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        conversation = [self._internal_prompt, formulation_request]

        self._user_says()
        self._global_conversations.append(self._internal_prompt.content)
        self._global_conversations.append(
            self._model_format(template_formulation)
        )  # noqa
        messages = self._prompt_format(conversation)
        self._chatbot_says()
        output = self._llm(messages=messages)
        self._llm_response = output.content
        self._global_conversations.append(self._llm_response)

        # Fill-in formulation response
        self._formulation_response = self._llm_response

        return output.content

    def _generate_code_standard(self) -> None:
        solve_request = HumanMessagePromptTemplate.from_template(
            template_standard_prompt
        )
        conversation = [self._internal_prompt, solve_request]

        self._user_says()
        self._global_conversations.append(self._internal_prompt.content)
        self._global_conversations.append(
            self._model_format(template_standard_prompt)
        )  # noqa
        messages = self._prompt_format(conversation)
        self._chatbot_says()
        output = self._llm(messages=messages)
        self._llm_response = output.content
        self._global_conversations.append(self._llm_response)

        try:
            self._solver_codes.append(output.content.split("```")[1][6:])
        except:
            self._solver_codes.append("raise Exception")

        self._data["code"] = self._solver_codes[-1]
        self.dump_code(path_to_code=os.path.join(self._path, "code.py"))

        return

    def _generate_code(self) -> None:
        """
        Let LLM generate code for the instance
        :return:
        """

        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        # Input the previous response
        formulation_response = AIMessage(content=self._formulation_response)
        codegen_request = HumanMessagePromptTemplate.from_template(template_codegen)
        conversation = [
            self._internal_prompt,
            formulation_request,
            formulation_response,
            codegen_request,
        ]

        self._user_says()
        self._model_format(template_codegen)
        self._global_conversations.append(self._model_format(template_codegen))  # noqa
        messages = self._prompt_format(conversation)
        self._chatbot_says()
        output = self._llm(messages=messages)
        self._llm_response = output.content
        self._global_conversations.append(self._llm_response)

        try:
            self._solver_codes.append(output.content.split("```")[1][6:])
        except IndexError:
            self._solver_codes.append(output.content)

        self._data["code"] = self._solver_codes[-1]
        self.dump_code(path_to_code=os.path.join(self._path, "code.py"))

        return output.content

    def _generate_test(self) -> None:
        """
        Let LLM generate test cases for the instance
        :return:
        """

        testgent_request = HumanMessagePromptTemplate.from_template(template_testgen)
        conversation = [self._internal_prompt, testgent_request]

        self._user_says()
        self._global_conversations.append(self._internal_prompt.content)
        self._global_conversations.append(self._model_format(template_testgen))  # noqa
        messages = self._prompt_format(conversation)

        self._chatbot_says()
        output = self._llm(messages=messages)
        self._llm_response = output.content
        self._global_conversations.append(self._llm_response)

        self._testgen_response = self._llm_response

        try:
            test_script = output.content.split("```")[1][6:]
        except IndexError:
            test_script = output.content

        with open(os.path.join(self._path, self._std_test), "w") as f:
            f.write(test_script)

        return

    def _run_and_fix(
        self, user_test_path: str, max_iter: int, syntax_only: bool
    ) -> None:
        """
        Call run-and-fix loop
        :param user_test_path: The path of test.py file
        :param max_iter: Maximum fixing iteration
        :return:
        """

        test_path = self._path
        current_path = os.getcwd()
        test_file = user_test_path
        os.chdir(test_path)

        # Make sure data.json exists in the current working directory
        if not os.path.isfile("data.json"):
            raise Exception(
                "data.json does not exist in the current working directory!"
            )  # noqa

        # Prepare test module
        if not syntax_only:
            spec = importlib.util.spec_from_file_location("test", test_file)  # noqa
            test = importlib.util.module_from_spec(spec)  # noqa
            spec.loader.exec_module(test)

        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        formulation_response = AIMessage(content=self._formulation_response)
        codefix_execution_request = HumanMessagePromptTemplate.from_template(
            template_codefix_execution
        )
        codefix_data_request = HumanMessagePromptTemplate.from_template(
            template_codefix_data
        )

        num_iter = 0

        while True:
            execution_ok = False
            num_iter += 1
            try:
                subprocess.run(
                    ["python", "code.py"], check=True, text=True, capture_output=True
                )
                execution_ok = True

                if syntax_only:
                    res = []
                else:
                    try:
                        res = test.run()
                    except:
                        self._log("==== Test script is invalid!")
                        self._iter = -1
                        break

                # if 'output.json' does not exist, then something is wrong with the input. print that
                # If we do not care about the validity of the logic
                if len(res) == 0:
                    break
                else:
                    self._log("==== Some tests failed!")
                    self._log("Test results: %s" % res)
                    raise subprocess.CalledProcessError(
                        returncode=1, cmd="python code.py", stderr="\n".join(res)
                    )

            except subprocess.CalledProcessError as e:
                self._errmsg = e.stderr

                if not execution_ok:
                    codefix_request = codefix_execution_request
                    template_codefix = template_codefix_execution
                else:
                    codefix_request = codefix_data_request
                    template_codefix = template_codefix_data

                self._log(
                    "Script failed and exited with an error code: %s" % e.returncode
                )
                self._log("Error message: %s" % e.stderr)
                self._log("==== Fixing the code...")

                conversation = [
                    self._internal_prompt,
                    formulation_request,
                    formulation_response,
                    codefix_request,
                ]

                # User asks: the model is not correct
                self._user_says()
                self._global_conversations.append(self._model_format(template_codefix))
                messages = self._prompt_format(conversation)

                self._log("Sending messages to the model...")
                output = self._llm(messages=messages)
                self._log("Heard back from the model!")
                self._llm_response = output.content

                # Get GPT response
                self._chatbot_says()
                self._global_conversations.append(self._llm_response)

                try:
                    self._solver_codes.append(output.content.split("```")[-2][6:])
                except:
                    self.dump_conversation()
                    raise Exception("Can't find code in the output of the LLM!")

                self._data["code"] = self._solver_codes[-1]
                self.dump_code()
                self.dump_conversation()

            if num_iter > max_iter:
                self._log("==== Giving up!")
                self._log(
                    "Please check problem and test files to make sure they are correct."
                )
                break

        self._iter = num_iter
        os.chdir(current_path)
        return

    def rephrase_problem(
        self,
        problem_path: str,
        problem_file: str,
        n_rephrase: int,
        output_path: str = None,
    ):
        """
        :param problem_path: Path to description.txt
        :param n_rephrase: Number of rephrased instances
        :param output_path: Path of output.
        :return:

        The rephrased instances will be named description-r1.txt, description-r2.txt,...,description-rn.txt
        """

        self._path = os.path.abspath(problem_path)
        file_name = os.path.split(os.path.abspath(problem_path))[1]
        self._prob_name = os.path.splitext(problem_file)[0]

        if output_path is None:
            output_path = self._path

        print(output_path)
        self._log("==== Connecting to chat bot...")
        self._connect_chatbot()
        self._problem_path = problem_path
        self._log("==== Collecting problem data...")
        self._get_problem_data()
        self._log("==== Rephrase the problem data %d times \n" % n_rephrase)

        rephrase_request = HumanMessagePromptTemplate.from_template(template_rephrase)
        conversation = [self._internal_prompt, rephrase_request]

        for k in range(n_rephrase):
            self._log("+ Rephrased problem %d" % (k + 1))
            self._user_says()
            self._global_conversations.append(self._model_format(template_rephrase))
            messages = self._prompt_format(conversation)
            output = self._llm(messages=messages)
            self._llm_response = output.content
            self._data["problem_info"] = self._llm_response
            self._chatbot_says()
            self._global_conversations.append(self._llm_response)
            generate_instance_template(
                self._data, os.path.join(output_path, f"{self._prob_name}-r{k+1}.txt")
            )

        return

    def _check_params(self, solve_mode: int, solve_params: Dict[str, bool]) -> None:
        """
        Manage the behavior of agent
        :param solve_mode:
        :param solve_params:
        :return:
        """
        self._params["COT"] = False
        self._params["Debug"] = False
        self._params["Test"] = False
        self._params["Human"] = False

        # Go through parameter interface
        if solve_params is not None:
            self._params = solve_params
        else:
            if solve_mode >= MODE_COT_ONLY:
                self._params["COT"] = True
            if solve_mode >= MODE_COT_DEBUG:
                self._params["Debug"] = True
            if solve_mode >= MODE_COT_DEBUG_TEST:
                self._params["Test"] = True
            if solve_mode >= MODE_COT_HUMAN:
                self._params["Human"] = True
                self._params["Test"] = False

        for key, val in self._params.items():
            self._log("{0}: {1}".format(key, val))
            self._global_conversations.append("{0}: {1}".format(key, val))

        self._iter = -1

        return

    def _benchmark(self, human_test_path: str) -> List[int]:
        """
        Use human-based testing script to do benchmark
        :return: Solution status

        This utility mimics the behavior of run_and_fix with only one iteration

        """

        test_path = self._path
        current_path = os.getcwd()
        abs_test_path = human_test_path
        os.chdir(test_path)

        # Make sure data.json exists in the current working directory
        if not os.path.isfile("data.json"):
            raise Exception(
                "data.json does not exist in the current working directory!"
            )  # noqa

        # Prepare test module
        spec = importlib.util.spec_from_file_location("test", abs_test_path)  # noqa
        test = importlib.util.module_from_spec(spec)  # noqa
        spec.loader.exec_module(test)

        # First evaluate validity of code syntax
        try:
            subprocess.run(
                ["python", "code.py"], check=True, text=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            os.chdir(current_path)
            return [STATUS_SYNTAX_ERROR, self._iter]

        res = test.run()

        if len(res) > 0:
            os.chdir(current_path)
            return [STATUS_LOGIC_ERROR, self._iter]

        os.chdir(current_path)
        return [STATUS_PASSED, self._iter]

    def solve_problem(
        self,
        problem_path: str,
        max_attempt: int,
        problem_file: str = None,
        bench_file: str = None,
        solver: str = None,
        solve_mode: int = MODE_COT_HUMAN,
        solve_params: Dict[str, bool] = None,
    ) -> List[int]:
        """
        Solve an OR instance using large language model
        :param problem_path: path to the problem folder
        :param problem_file: replace description.txt by this
        :param bench_file: test-human.txt
        :param max_attempt: maximum attempts to fix a instance
        :param solver: the solver used
        :param solve_mode: strategy for solving the model
        :param solve_params: detailed strategy switch
        :return:

        ================================================================================
         Further comments on solve_mode
         To facilitate the test, we currently implement the following behaviors of solvers for the agent
         These behaviors matches the experiments presented in the paper

            Mode/Behavior         Chain of thought  Debugging syntax  Testing logic   Human logic \n
            1. Standard Prompt Only     False           False           False           False     \n
            2. CoT Only                 True            False           False           False     \n
            3. CoT + Debug              True            True            False           False     \n
            4. CoT + Debug + Test       True            True            True            False     \n
            5. CoT + Debug + Human      True            True            True            True      \n

        Note that
        - Augmentation can be combined with arbitrary mode and is independently implemented
        - Choice between human and testing logic is managed by choosing different testing scripts
        - If solve_params is provided, it will override solve_mode

        """

        # Check parameters
        self._check_params(solve_mode, solve_params)

        self._path = problem_path
        self._prob_name = problem_path

        self._std_format = problem_file
        self._std_log = "%s-%d.log" % (
            os.path.splitext(self._std_format)[0],
            solve_mode,
        )
        self._std_test = "test-%s.py" % os.path.splitext(self._std_format)[0]

        os.system("rm %s" % os.path.join(self._path, "output.json"))

        if solver is not None:
            self._solver = solver

        # Common step aaug all the parameter settings
        self._log("==== Connecting to chat bot...")
        self._connect_chatbot()
        self._problem_path = problem_path

        self._log("==== Collecting problem data...")
        self._get_problem_data()

        # Use pure prompt or chain of thought ?
        if not self._params["COT"]:
            # If not using COT,
            self._log("==== Generating the code using standard prompting...")
            self._generate_code_standard()

        else:
            self._log("==== Generating the formulation...")
            res = self._generate_formulation()

            if self.verbose:
                print(res)
                self.dump_conversation(os.path.join(self._path, self._std_log))

            self._log("==== Generating the code...")
            res = self._generate_code()

            if self.verbose:
                print(res)
                self.dump_conversation(os.path.join(self._path, self._std_log))

        # If we debug but do not do automatic test
        if (
            self._params["Debug"]
            and not self._params["Test"]
            and not self._params["Human"]
        ):
            self._run_and_fix(
                user_test_path=self._std_test,
                max_iter=max_attempt,
                syntax_only=True,
            )

        # If we test logic, then debug must be turned on
        if self._params["Test"]:
            self._log("==== Generating the test...")
            self._generate_test()
            self._run_and_fix(
                user_test_path=self._std_test,
                max_iter=max_attempt,
                syntax_only=False,
            )

        # If we enable human logic to test. Both debug and test are turned on
        if self._params["Human"]:
            self._run_and_fix(
                user_test_path=bench_file, max_iter=max_attempt, syntax_only=False
            )

        self._log("==== Done with generation!")
        # Get solution status
        self._log("==== Benchmarking the solution...")

        self.dump_conversation(os.path.join(self._path, self._std_log))

        return self._benchmark(bench_file)

    def print_problem(self) -> None:
        for key, val in self._data.items():
            self._log("------------------------")
            self._log("%s" % key)
            self._log("------------------------")
            self._log("%s \n" % val)
        return

    def dump_code(self, path_to_code: str = None) -> None:
        if self._data["code"] is None:
            raise ValueError("No code to dump!")

        if path_to_code is None:
            path_to_code = "code.py"
        with open(path_to_code, "w") as f:
            f.write(self._solver_codes[-1])

    def dump_conversation(self, path_to_conversation: str = None) -> None:
        if path_to_conversation is None:
            path_to_conversation = self._std_log

        print("Dumping conversation to %s" % path_to_conversation)

        with open(path_to_conversation, "w") as f:
            f.write("\n".join(self._global_conversations))


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name")
parser.add_argument(
    "--prob",
    type=str,
    default="./datasets/introduction_to_linear_optimization/problem_15",
    help="Problem path",
)
parser.add_argument(
    "--stdfname", type=str, default="description.txt", help="Description file name"
)
parser.add_argument("--maxtry", type=int, default=1, help="Maximum attempts")
parser.add_argument(
    "--human", type=str, default="test-human.py", help="Human test file"
)
parser.add_argument(
    "--aug",
    type=int,
    default=0,
    help="Number of augmentations. Uses the rephrases generated by using the --rephrase parameter.",
)
parser.add_argument(
    "--solver", type=str, default="gurobi", help="Solver name (cvxpy/gurobi)"
)
parser.add_argument(
    "--mode",
    type=int,
    default=105,
    help="102: Prompt, 103: Prompt + Debug, 104: Prompt + Debug + AutoTest, 105: Prompt + Debug + Human Test",
)
parser.add_argument(
    "--rephrase",
    type=int,
    default=0,
    help="Number of rephrases. If more than 0, the agent will only generate rephrases of the problem and will NOT solve the problem. The rephrased instances can then later be used by setting the aug parameter to the number of rephrases.",
)
parser.add_argument("--verbose", type=bool, default=False, help="Verbose mode")

if __name__ == "__main__":
    args = parser.parse_args()
    llm_model = args.model
    prob = args.prob
    std_file = args.stdfname
    mode = args.mode

    """
    MODE_STANDARD_PROMPT = 101
    MODE_COT_ONLY = 102
    MODE_COT_DEBUG = 103
    MODE_COT_DEBUG_TEST = 104
    MODE_COT_HUMAN = 105
    """

    or_agent = GPT4OR(
        model_name=llm_model,
        solver=args.solver,
        api_key=api_keys[0],
        verbose=args.verbose,
    )

    if args.rephrase > 0:
        or_agent.rephrase_problem(
            problem_path=prob, problem_file=std_file, n_rephrase=args.rephrase
        )
    else:
        try:
            status = or_agent.solve_problem(
                problem_path=prob,
                problem_file=std_file,
                bench_file=args.human,
                max_attempt=args.maxtry,
                solver=args.solver,
                solve_mode=mode,
                solve_params=None,
            )

            print("Status: ", status)
        except Exception as e:
            raise e

    or_agent.dump_conversation(os.path.join(or_agent._path, or_agent._std_log))
