import openai
from utils import get_response
import os
import json
import traceback
import subprocess

prompt_template = """
You are an expert operations research analyst. Your task is to generate Gurobi code to solve the following optimization problem:

{problem_description}

The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
First, reason about the problem and model it. Then, generate gurobipy code to solve it. Put the code between two '=====' lines, like this:

=====
import ...
...
=====

- The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
- Generate the complete code, including the model definition, variables, constraints, objective function, and optimization. It must be runnable.
- Do not generate anything after the second '====='.
- Take a deep breath and think step by step.
"""

reflection_template = """
You are an expert operations research analyst. You have been given the task to generate Gurobi code to solve an optimization problem. You have generated the following Gurobi code:

{generated_code}

You have been updating the code for these errors (the last one is the most recent one):

{feedback}

Based on this feedback, suggest improvements to the Gurobi code.
First, reason about the problem and model it. Then, generate gurobipy code to solve it. Put the code between two '=====' lines, like this:

=====
import ...
...
=====

- The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
- Generate the complete code, including the model definition, variables, constraints, objective function, and optimization. It must be runnable.
- Do not generate anything after the second '====='.
- Take a deep breath and think step by step.
"""


def extract_code(text):
    ind1 = text.find("=====")
    ind2 = text.find("=====", ind1 + 5)

    code = text[ind1 + 5 : ind2].strip()
    code = code.replace("```python", "").replace("```", "").strip()

    return code


def execute_code(file_path):
    try:
        # Using Python's subprocess to execute the code as a separate process
        result = subprocess.run(
            ["python", file_path], capture_output=True, text=True, check=True
        )
        # save result in a file
        with open(
            os.path.join(os.path.dirname(file_path), "ref_optimal_value.txt"), "w"
        ) as f:
            f.write(f"Optimal Revenue: {result.stdout}\n")
        return result.stdout, "Success"
    except subprocess.CalledProcessError as e:
        return e.stderr, "Error"


def main(problem_description, dir, max_iter=3):
    feedback = ""
    current_prompt = prompt_template.format(problem_description=problem_description)

    print(current_prompt)
    print("====================\n\n\n\n")
    for iteration in range(max_iter):
        response = get_response(current_prompt, model="llama3-70b-8192")
        code = extract_code(response)

        # Save the code to a file
        code_filename = f"generated_code_{iteration}.py"
        code_file_path = os.path.join(dir, "ref_codes", code_filename)
        with open(code_file_path, "w") as f:
            f.write(code)

        # Execute the code
        output, status = execute_code(code_file_path)

        # Save error output (if any)
        error_filename = f"error_{iteration}.txt"
        if status == "Error":
            with open(os.path.join(dir, "ref_codes", error_filename), "w") as f:
                f.write(output)

        # Print status and update the prompt if needed
        if status == "Success":
            print("Code executed successfully. Output:\n", output)
            break
        else:
            feedback += "\n" + output
            current_prompt = reflection_template.format(
                generated_code=code, feedback=feedback
            )
            print(f"Iteration {iteration + 1}: Error encountered. Debugging...")
    else:
        print("Max iterations reached with errors remaining.")


if __name__ == "__main__":
    dir = "data/nlp4lp/train-dev/1"
    if not os.path.exists(os.path.join(dir, "ref_codes")):
        os.makedirs(os.path.join(dir, "ref_codes"))

    with open(os.path.join(dir, "desc.txt"), "r") as f:
        desc = f.read()

    with open(os.path.join(dir, "data.json"), "r") as f:
        data = json.load(f)

    desc = desc + "\n" + json.dumps(data, indent=4)
    main(desc, dir, max_iter=3)
