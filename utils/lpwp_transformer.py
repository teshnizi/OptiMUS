import openai
import json
import os
import sys
import re

from misc import get_openai_client

prompt_template = """
You are an expert mathematical modeler and an optimization professor at a top university. Your task is to read a given optimization problem, and re-write it in a text-book format.

Here is the problem:

-----
{description}
-----

you should go through the statements one by one, and identify and separate the parameters of the problem, and put them in a json file and a string in this format:

=====
{{
    "parameters": {{
        "definition": str
        "symbol": str
        "value": number
        "shape": [str]
    }}
}}
=====
*updated description* 
=====

Where 
- "definition" is the the definition of the parameter.
- "symbol" is the mathematical symbol you want to use to represent the parameter.
- "value" a float or int, representing the numerical value of the variable (use 0.33 instead of 1/3.0)
- "shape" is a possibly empty list of string representing the dimensions of the parameter in terms of other parameters.
- *updated description* is the original description of the problem with its numbers replaced with the corresponding variable symbols.

Here is an example:

*** input ***

A firm produces M different goods using 10 different raw materials. The firm has available_{{i}} of raw material i available. Good j requires req_{{i,j}} units of material i per unit produced. Good j results in a revenue of price_{{j}} per unit produced. How much of each good should the firm produce in order to maximize its total revenue?


*** output ***

=====
{{
    "parameters": [
    {{
        "definition": "Number of different goods produced",
        "symbol": "M",
        "value": ""
        "shape": []
    }},
    {{
        "definition": "Number of different raw materials",
        "symbol": "N",
        "value": 10,
        "shape": []
    }}
    {{
        "definition": "Amount of raw material i available",
        "symbol": "Available",
        "value": "",
        "shape": ["N"]
    }},
    {{
        "definition": "Amount of raw material i required to produce one unit of good j",
        "symbol": "Required",
        "value": "",
        "shape": ["N", "M"]
    }},
    {{
        "definition": "Price of good j",
        "symbol": "Price",
        "value": "",
        "shape": ["M"]
    }}
    ]
}}
=====
A firm produces \\param{{M}} different goods using \\param{{N}} different raw materials. The firm has \\param{{Available}} of raw material i available. Good j requires \\param{{Required}} units of material i per unit produced. Good j results in a revenue of \\param{{Price}} per unit produced. How much of each good should the firm produce in order to maximize its total revenue?
=====

- Only generate the json file and the updated description, and do not generate anything else.
- Include the ===== lines in the output.
- Avoid using fractions and use floating point numbers instead (0.2 instead of 1/5)
- Note that indices are not separate parameters.
- Feel free to define new symbols for parameters that do not have a symbol.
- Use CamelCase and full words for symbols, and don't include the indices (e.g. MaxColor instead of maxColor or max_color or maxcolor or MaxCol or MaxColor_i or MaxColor_{{i}})
- Use single capital letters for symbols that represent dimensions for indices of other parameters (e.g. N, M, etc.)
- Note that parameters are known values upon which the model is built, and they do not change during the optimization process.  However, variables are the unknowns that the optimization process seeks to solve. DO NOT include variables in the parameters list!
- Make sure you include all the parameters in the updated problem description.

Take a deep breath and tackle the problem step by step.
"""

snop_prompt_template = """
You are an expert mathematical modeler and an optimization professor at a top university. Your task is to transform an optimization problem given in some format into the other given format.

Here is the problem:

-----
#### PROBLEM TYPE
{PROBLEM_TYPE}

#### INPUT FORMAT 
{INPUT_FORMAT}

#### OBJECTIVE
{OBJECTIVE}
-----

This format has three fields: PROBLEM INFO, INPUT FORMAT and OBJECTIVE, where 

1. PROBLEM INFO gives the basic description of the problem. The problem parameters are marked by \\var
2. INPUT FORMAT is pseudo JSON format in which the problem data parameters will be stored
3. OBJECTIVE defines the optimization objective

you should go through the statements one by one, and identify and separate the parameters of the problem, and put them in a json file and a string in this format:

=====
{{
    "parameters": {{
        "definition": str
        "symbol": str
        "value": number
        "shape": [str]
    }}
}}
=====
*updated description* 
=====

Where 
- "definition" is the the definition of the parameter.
- "symbol" is the mathematical symbol you want to use to represent the parameter.
- "value" a float or int, representing the numerical value of the variable (use 0.33 instead of 1/3.0)
- "shape" is a possibly empty list of string representing the dimensions of the parameter in terms of other parameters.
- *updated description* is the original description of the problem with its numbers replaced with the corresponding variable symbols.

Here is an example:

*** input ***

PROBLEM INFO:

- A firm produces \\var{{M}} different goods using \\var{{N}} different raw materials.
- The firm has \\var{{available_{{i}}}} of raw material \\var{{i}} available.
- Good \\var{{j}} requires \\var{{req_{{i,j}}}} units of material \\var{{i}} per unit produced.
- Good \\var{{j}} results in a revenue of \\var{{price_j}} per unit produced.


INPUT FORMAT: 

{{
    "available": [available_{{i}} for i in 1, ..., N]
    "requirements": [[req_{{i,j}} for i in 1, ..., N] for j in 1, ..., M],
    "prices": [price_{{j}} for j in 1, ..., M]
}}

OBJECTIVE: How much of each good should the firm produce in order to maximize its total revenue?

*** output ***

=====
{{
    "parameters": [
    {{
        "definition": "Number of different goods produced",
        "symbol": "M",
        "value": ""
        "shape": []
    }},
    {{
        "definition": "Number of different raw materials",
        "symbol": "N",
        "value": 10,
        "shape": []
    }}
    {{
        "definition": "Amount of raw material i available",
        "symbol": "Available",
        "value": "",
        "shape": ["N"]
    }},
    {{
        "definition": "Amount of raw material i required to produce one unit of good j",
        "symbol": "Required",
        "value": "",
        "shape": ["N", "M"]
    }},
    {{
        "definition": "Price of good j",
        "symbol": "Price",
        "value": "",
        "shape": ["M"]
    }}
    ]
}}
=====
A firm produces \\var{{M}} different goods using \\var{{N}} different raw materials. The firm has \\var{{Available}} of raw material i available. Good j requires \\var{{Required}} units of material i per unit produced. Good j results in a revenue of \\var{{Price}} per unit produced. How much of each good should the firm produce in order to maximize its total revenue?
=====

- Only generate the json file and the updated description, and do not generate anything else.
- Include the ===== lines in the output.
- Avoid using fractions and use floating point numbers instead (0.2 instead of 1/5)
- Note that indices are not separate parameters.
- Feel free to define new symbols for parameters that do not have a symbol.
- Use CamelCase and full words for symbols, and don't include the indices (e.g. MaxColor instead of maxColor or max_color or maxcolor or MaxCol or MaxColor_i or MaxColor_{{i}})
- Use single capital letters for symbols that represent dimensions for indices of other parameters (e.g. N, M, etc.)
- Note that parameters are known values upon which the model is built, and they do not change during the optimization process.  However, variables are the unknowns that the optimization process seeks to solve. DO NOT include variables in the parameters list!
- Make sure you include all the parameters in the updated problem description.
- Keep \\var around the parameters in the description

Take a deep breath and tackle the problem step by step.
"""


def transform_lpwp_instance(folder_dir: str, client):
    """Clean the LPWP instance in the given folder.

    Args:
        folder_dir (str): the folder that contains the LPWP instance
    """

    with open(folder_dir + "/description.txt", "r") as f:
        description = f.read()

    prompt = prompt_template.format(description=description)

    print(prompt)

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    output = completion.choices[0].message.content

    print("-=" * 10)
    print(output)
    print("-=" * 10)

    output = output.split("=====")
    output = [x.strip() for x in output if len(x.strip()) > 0]

    output_json = output[0]
    output_desc = output[1]

    if "```json" in output_json:
        # delete until the first '```json'
        output_json = output_json[output_json.find("```json") + 7 :]

        # delete until the last '```'
        output_json = output_json[: output_json.rfind("```")]

    print(output_json)

    with open("tmp.json", "w") as f:
        f.write(output_json)

    update = json.loads(output_json)

    for param in update["parameters"]:
        if "_" in param["symbol"]:
            new_symbol = param["symbol"].replace("_", "")
            output_desc = output_desc.replace(param["symbol"], new_symbol)
            param["symbol"] = new_symbol

    update["description"] = output_desc

    return update


def read_problem_from_entire_file(problem_file: str):
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


def transform_nlp4lp_instance(snop: str, client):
    """
    Clean the NLP4LP dataset from a given snop
    :param snop:
    :param client:
    :return:
    """

    prob = read_problem_from_entire_file(snop)
    prompt = snop_prompt_template.format(
        PROBLEM_TYPE=prob["problem_type"],
        INPUT_FORMAT=prob["input_format"],
        OBJECTIVE=prob["objective_info"],
    )
    print(prompt)

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    output = completion.choices[0].message.content

    print("-=" * 10)
    print(output)
    print("-=" * 10)

    output = output.split("=====")
    output = [x.strip() for x in output if len(x.strip()) > 0]

    output_json = output[0]
    output_desc = output[1]

    if "```json" in output_json:
        # delete until the first '```json'
        output_json = output_json[output_json.find("```json") + 7 :]

        # delete until the last '```'
        output_json = output_json[: output_json.rfind("```")]

    print(output_json)

    with open("tmp.json", "w") as f:
        f.write(output_json)

    update = json.loads(output_json)

    for param in update["parameters"]:
        if "_" in param["symbol"]:
            new_symbol = param["symbol"].replace("_", "")
            output_desc = output_desc.replace(param["symbol"], new_symbol)
            param["symbol"] = new_symbol

    update["description"] = output_desc

    return update


if __name__ == "__main__":
    client = get_openai_client()

    path = "data/nl4opt/LPWP"

    for i in range(1, 288):
        print(i)
        if os.path.exists(os.path.join(path, f"prob_{i}/input.json")):
            continue
        try:
            json_file = transform_lpwp_instance(os.path.join(path, f"prob_{i}"), client)
            json_str = json.dumps(json_file, indent=4)
            with open(os.path.join(path, f"prob_{i}/input.json"), "w+") as f:
                f.write(json_str)
        except Exception as e:
            print(f"Error in {i}: {e}")
