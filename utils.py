import os
import subprocess
import importlib

from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessage
from langchain.prompts import ChatPromptTemplate

import re
from typing import List, Dict, Any, Union


def read_problem_from_entire_file(problem_file: str) -> Dict[str, str]:
    """
    Split the problem in different function blocks
    :param problem_file: target .txt model file
    :return:
    """

    problem_type_symbol = "PROBLEM TYPE:"
    problem_info_symbol = "PROBLEM INFO:"
    input_format_symbol = "INPUT FORMAT:"
    objective_symbol = "OBJECTIVE:"
    output_info_symbol = "OUTPUT INFO:"
    output_format_symbol = "OUTPUT FORMAT:"

    # Get regular expression for matching the files
    reg_expr = r"^%s|^%s|^%s|^%s|^%s|^%s" % (
        problem_type_symbol,
        problem_info_symbol,
        input_format_symbol,
        objective_symbol,
        output_info_symbol,
        output_format_symbol,
    )

    with open(problem_file, "r") as f:
        data = f.read()

    matching_positions = []
    split_data = data.splitlines()

    for num_line, line in enumerate(split_data, 1):
        if re.search(reg_expr, line):
            matching_positions.append(num_line - 1)

    if len(matching_positions) < 6:
        raise RuntimeError(f"The description file is incomplete")

    # Collect data
    problem_type = (
        split_data[matching_positions[0]].split(problem_type_symbol)[1].lstrip()
    )
    problem_info = "\n".join(
        split_data[line]
        for line in range(matching_positions[1] + 1, matching_positions[2])
    )
    input_format = "\n".join(
        split_data[line]
        for line in range(matching_positions[2] + 1, matching_positions[3])
    )
    objective_info = (
        split_data[matching_positions[3]].split(objective_symbol)[1].lstrip()
    )
    output_info = "\n".join(
        split_data[line]
        for line in range(matching_positions[4] + 1, matching_positions[5])
    )
    output_format = "\n".join(
        split_data[line] for line in range(matching_positions[5] + 1, len(split_data))
    )

    return {
        "problem_type": problem_type,
        "problem_info": problem_info,
        "input_format": input_format,
        "objective_info": objective_info,
        "output_format": output_format,
        "output_info": output_info,
        "code": "",
    }


def generate_instance_template(output_data: Dict[str, str], output_files: str) -> None:
    file = open(output_files, "w")
    file.write("PROBLEM TYPE: %s \n" % output_data["problem_type"])
    file.write("PROBLEM INFO: \n\n")
    file.write("%s \n\n" % output_data["problem_info"])
    file.write("INPUT FORMAT: \n")
    file.write("%s \n" % output_data["input_format"])
    file.write("OBJECTIVE: %s \n\n" % output_data["objective_info"])
    file.write("OUTPUT INFO: \n")
    file.write("%s \n" % output_data["output_info"])
    file.write("OUTPUT FORMAT: \n")
    file.write("%s \n" % output_data["output_format"])
    file.close()

    return


str_get_all_json_keys = """

def parse_json(json_file, keys):

    if isinstance(json_file, list):
        for v in json_file:
            parse_json(v, keys)
    elif isinstance(json_file, dict):
        for k, v in json_file.items():
            if isinstance(v, dict) or isinstance(v, list):
                parse_json(v, keys)
            if k not in keys:
                keys.append(k)                
"""


def get_initial_test_script(output_format: str):
    json_fields = []
    output_inlines = output_format.splitlines()
    for num_line in range(1, len(output_inlines) - 1):
        line = output_inlines[num_line]
        field = re.findall(r'"([^"]*)"', line)
        if len(field) > 0:
            json_fields.append(field[0])

    # file = open(test_path, "w")
    script = ""

    # Generate header of the verification code
    script += "import json\n"
    script += "\neps = 1e-06\n"
    script += str_get_all_json_keys
    script += "\n\ndef run():\n\n"
    script += "    with open('data.json', 'r') as f:\n"
    script += "        data = json.load(f)\n\n"
    script += "    with open('output.json', 'r') as f:\n"
    script += "        output = json.load(f)\n\n"
    script += "    all_json_keys = []\n"
    script += "    parse_json(output, all_json_keys)\n\n"
    script += "    error_list = []\n\n"

    # Check if the outputs are available
    for num_field in range(len(json_fields)):
        script += """    if not "%s" in all_json_keys:\n""" % json_fields[num_field]
        script += (
            """        error_list.append("The output field '%s' is missing")\n\n"""
            % json_fields[num_field]
        )

    script += "\n\n    #---------------------------------------------\n"
    script += "    # Write problem specific testing code here\n"
    script += "    #---------------------------------------------\n\n"

    script += "    return error_list\n\n"

    script += "if __name__ == '__main__': \n"
    script += "    print(run())\n"

    return script


def get_solver_instruction(solver: str) -> str:
    if solver == "cvxpy":
        return "- cvxpy.sum takes a list as input, and not a generator"
    elif solver == "gurobi":
        return "- if problem data is presented in percentage (%), do not forget to preprocess it"
    else:
        return ""


def read_problem_from_file(problem_path):
    problem_path = os.path.join(problem_path, "description")

    with open(os.path.join(problem_path, "problem_info.txt"), "r") as f:
        problem_info = f.read()

    with open(os.path.join(problem_path, "input_format.txt"), "r") as f:
        input_format = f.read()

    with open(os.path.join(problem_path, "objective.txt"), "r") as f:
        objective = f.read()

    with open(os.path.join(problem_path, "output_format.txt"), "r") as f:
        output_format = f.read()

    with open(os.path.join(problem_path, "output_info.txt"), "r") as f:
        output_info = f.read()

    return {
        "problem_info": problem_info,
        "input_format": input_format,
        "objective": objective,
        "output_format": output_format,
        "output_info": output_info,
    }


def get_templates():
    template_path = os.path.join(os.path.split(__file__)[0], "templates")

    with open(os.path.join(template_path, "template_formulation.txt")) as f:
        template_formulation = f.read()

    with open(os.path.join(template_path, "template_codegen.txt")) as f:
        template_codegen = f.read()

    with open(os.path.join(template_path, "template_codefix_execution.txt")) as f:
        template_codefix_execution = f.read()

    with open(os.path.join(template_path, "template_codefix_data.txt")) as f:
        template_codefix_data = f.read()

    with open(os.path.join(template_path, "template_doublecheck.txt")) as f:
        template_doublecheck = f.read()

    with open(os.path.join(template_path, "template_rephrase.txt")) as f:
        template_rephrase = f.read()

    with open(os.path.join(template_path, "template_testgen.txt")) as f:
        template_testgen = f.read()

    with open(os.path.join(template_path, "template_standard_prompt.txt")) as f:
        template_standard_prompt = f.read()

    return {
        "formulation": template_formulation,
        "codegen": template_codegen,
        "codefix_execution": template_codefix_execution,
        "codefix_data": template_codefix_data,
        "doublecheck": template_doublecheck,
        "rephrase": template_rephrase,
        "testgen": template_testgen,
        "standard_prompt": template_standard_prompt,
    }


def generate_formulation(
    llm, templates, system_message, problem, problem_path, file_name="formulation.txt"
):
    formulation_request = HumanMessagePromptTemplate.from_template(
        templates["formulation"]
    )

    conversation = [system_message, formulation_request]

    messages = ChatPromptTemplate.from_messages(messages=conversation).format_messages(
        PROBLEM_INFO=problem["problem_info"],
        input_format=problem["input_format"],
        OBJECTIVE=problem["objective"],
        OUTPUT_INFO=problem["output_info"],
        OUTPUT_FORMAT=problem["output_format"],
    )

    output = llm(messages=messages)

    with open(os.path.join(problem_path, file_name), "w") as f:
        f.write(output.content)


def generate_code(
    llm,
    templates,
    system_message,
    problem,
    problem_path,
    file_name="code.py",
    double_check=True,
):
    formulation_request = HumanMessagePromptTemplate.from_template(
        templates["formulation"]
    )

    with open(os.path.join(problem_path, "formulation.txt"), "r") as f:
        formulation = f.read()

    formulation_response = AIMessage(content=formulation)

    codegen_request = HumanMessagePromptTemplate.from_template(templates["codegen"])

    conversation = [
        system_message,
        formulation_request,
        formulation_response,
        codegen_request,
    ]

    messages = ChatPromptTemplate.from_messages(messages=conversation).format_messages(
        PROBLEM_INFO=problem["problem_info"],
        input_format=problem["input_format"],
        OBJECTIVE=problem["objective"],
        OUTPUT_INFO=problem["output_info"],
        OUTPUT_FORMAT=problem["output_format"],
    )

    output = llm(messages=messages)
    code = output.content.split("```")[1][6:]

    if double_check:
        codegen_response = AIMessage(content=output.content)

        doublecheck_request = HumanMessagePromptTemplate.from_template(
            templates["doublecheck"]
        )

        conversation = [
            system_message,
            formulation_request,
            formulation_response,
            codegen_request,
            codegen_response,
            doublecheck_request,
        ]

        messages = ChatPromptTemplate.from_messages(
            messages=conversation
        ).format_messages(
            PROBLEM_INFO=problem["problem_info"],
            input_format=problem["input_format"],
            OBJECTIVE=problem["objective"],
            OUTPUT_INFO=problem["output_info"],
            OUTPUT_FORMAT=problem["output_format"],
        )

        output = llm(messages=messages)

        if not "--- OK ---" in output.content:
            code = output.content.split("```")[1][6:]

    with open(os.path.join(problem_path, file_name), "w") as f:
        f.write(code)


def run_and_fix_code(llm, templates, system_message, problem, problem_path):
    # switch the working directory to the problem directory
    original_directory = os.getcwd()
    os.chdir(problem_path)

    # import the test module
    test_module_path = os.path.join(os.getcwd(), "test.py")

    spec = importlib.util.spec_from_file_location("test", test_module_path)
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)

    # read messages from file
    formulation_request = HumanMessagePromptTemplate.from_template(
        templates["formulation"]
    )

    with open("code.py", "r") as f:
        code = f.read()

    with open("formulation.txt", "r") as f:
        formulation = f.read()
    formulation_response = AIMessage(content=formulation)

    codefix_request = HumanMessagePromptTemplate.from_template(templates["codefix"])

    # run the fixing loop
    iters = 0
    while True:
        iters += 1
        try:
            # Run the script and capture the standard output and standard error
            completed_process = subprocess.run(
                ["python", "code.py"], check=True, text=True, capture_output=True
            )
            print("Code ran successfully!")
            res = test.run()
            if res == "OK":
                print("==== All tests passed!")
                break
            else:
                print("==== Some tests failed!")
                print("Test results:", res)
                # manually throw and error with the res as the error message
                raise subprocess.CalledProcessError(
                    returncode=1, cmd="python code.py", stderr=res
                )

        except subprocess.CalledProcessError as e:
            print("Script failed and exited with an error code:", e.returncode)
            print("Error message:", e.stderr)
            print("==== Fixing the code...")
            conversation = [
                system_message,
                formulation_request,
                formulation_response,
                codefix_request,
            ]

            messages = ChatPromptTemplate.from_messages(
                messages=conversation
            ).format_messages(
                PROBLEM_INFO=problem["problem_info"],
                input_format=problem["input_format"],
                OBJECTIVE=problem["objective"],
                OUTPUT_INFO=problem["output_info"],
                OUTPUT_FORMAT=problem["output_format"],
                ERROR_MESSAGE=e.stderr,
                CODE=code,
            )

            print("Sending messages to the model...")
            output = llm(messages=messages)
            print("Heard back from the model!")

            code = output.content.split("```")[-2][6:]
            with open("code.py", "w") as f:
                f.write(code)

        if iters > 5:
            print("==== Giving up!")
            print("Please check problem and test files to make sure they are correct.")
            break

    # Change back to the original working directory
    os.chdir(original_directory)
