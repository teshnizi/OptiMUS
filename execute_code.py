import os
import numpy as np
import json
from utils import get_response
import subprocess

debug_template = """
You are an Operations Research consultant hired to address optimization issues for a company. Below is the problem description and the problematic code, followed by the error it produces:

Problem Description:
{description}

Problematic Code:
{code}

Error Message:
{error}

Your task is to debug the code. Begin by assessing the situation, then provide the corrected code in the following format:

=====
import ...
...

=====

- Ensure no output follows the closing ===== line.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def extract_code(text):
    ind_1 = text.find("=====")
    ind_2 = text.find("=====", ind_1 + 1)
    code = text[ind_1 + 5 : ind_2].strip()
    code = code.replace("```python", "").replace("```", "").strip()

    return code


def execute_code(dir, code_filename):
    try:
        code_path = os.path.join(dir, code_filename)
        # Using Python's subprocess to execute the code as a separate process
        result = subprocess.run(
            ["python", code_filename],
            capture_output=True,
            text=True,
            check=True,
            cwd=dir,
        )
        # save result in a file
        with open(os.path.join(dir, "code_output.txt"), "w") as f:
            f.write(f"Optimal Revenue: {result.stdout}\n")
        return result.stdout, "Success"
    except subprocess.CalledProcessError as e:
        return e.stderr, "Error"


def execute_and_debug(state, dir, model, logger, max_tries=3):

    code_filename = "code.py"
    with open(os.path.join(dir, code_filename), "r") as f:
        code = f.read()

    for iteration in range(max_tries):

        # Execute the code
        output, status = execute_code(dir, code_filename)

        # Print status and update the prompt if needed
        if status == "Success":
            logger.log("Code executed successfully. Output:\n" + output)
            break
        else:
            error_filename = f"error_{iteration}.txt"
            with open(os.path.join(dir, error_filename), "w") as f:
                f.write(output)

            p = debug_template.format(
                description=state["description"], code=code, error=output
            )
            logger.log(f"Iteration {iteration + 1}: Error encountered. Debugging...")
            logger.log(p)
            logger.log("==========\n\n\n\n")

            response = get_response(p, model=model)
            logger.log("Response received.")
            logger.log(response)
            logger.log("==========\n\n\n\n")

            code = extract_code(response)
            code_filename = f"code_{iteration + 1}.py"
            code_file_path = os.path.join(dir, code_filename)
            with open(code_file_path, "w") as f:
                f.write(code)
            logger.log(f"Iteration {iteration + 1}: Error encountered. Debugging...")
    else:
        logger.log("Max iterations reached with errors remaining.")
