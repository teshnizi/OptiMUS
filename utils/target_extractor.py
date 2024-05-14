import openai
import json
import os
from misc import get_openai_client

prompt_templates = [
    """
You are an optimization expert with a focus on mathematical and operational problems. Your task is to interpret a given optimization problem description, understand its nuances, and convert it into a structured standard form.

Upon receiving a problem description (with parameters marked as \param{{param_name}}), you should:

1. Carefully analyze and comprehend the problem.
2. Clearly define the background and context of the problem.
3. Identify and list all constraints, including any implicit ones like non-negativity.
4. Determine the primary objective of the problem.
5. If any ambiguities exist in the description, note them specifically.
6. Preferences are not constraints. Do not include them in the list of constraints.
7. Statements that simply define exact values of parameters are not constraints. Do not include them in the list of constraints (e.g., "The cost of producing an X is Y", or "Each X has a size of Y").
8. Statements that define bounds are constraints. Include them in the list of constraints (e.g., "The cost of producing an X is at most Y", or "Each X has a size of at least Y").

Produce a JSON file encapsulating this information in the following format:

{{
    "background": "String detailing the problem's background",
    "constraints": ["List", "Of", "All", "Constraints"],
    "objective": "The primary objective to be achieved"
}}


In case of ambiguities, use this format:

{{
    "ambiguities": ["List", "Of", "Identified", "Ambiguities"]
}}

Remember to keep each constraint separate and explicit. Do not merge different constraints into a single line.

Here is an example to illustrate:

*** Input ***

An office supply company manufactures two types of printers: color and black and white. Each type is produced by separate teams. The color printer team can produce up to \param{{MaxColor}} units per day, while the black and white team can produce up to \param{{MaxBW}} units per day. Both teams use a common machine for installing paper trays, which has a maximum daily capacity of \param{{MaxTotal}} printers. Each color printer yields a profit of \param{{ProfitColor}}, and each black and white printer yields \param{{ProfitBW}}. Determine the optimal production mix to maximize profit.

*** Output ***

{{
    "background": "An office supply company manufactures color and black and white printers, each by different teams using a shared resource."
    
    "constraints": [
        "Number of color printers is non-negative",
        "Number of black and white printers is non-negative",
        "Up to MaxColor color printers can be produced per day",
        "Up to MaxBW black and white printers can be produced per day",
        "A total of up to MaxTotal printers can be produced per day"
    ],
    "objective": "Maximize the company's profit from printer production"
}}



- Take a deep breath and approach this task methodically, step by step.
- First read and understand the problem description carefully. Then, generate the JSON file. Do not generate anything after the JSON file.

Here is the problem description: 

{description}

""",
    """
    
You are an optimization expert. I have this problem description:

{description}

Someone has extracted the following list of constraints from the description:

{constraints_json}

Your task is to go through the extracted constraints, and for each one of them, do the following:

1. If the statement is not actually a constraint or objective, mark it as "invalid".
2. Otherwise, find the relevant section of the problem description that the statement is referring to.

Generate a json file with the following structure:

{{
    "constraints": [
        {{
            "definition": "[Definition of the constraint]",
            "reasoning": "[Explanation of why the constraint is valid/invalid]",
            "status": "valid/invalid/redundant",
            "relevant_section": "[Section of the problem description that the constraint is referring to, or 'none' if the constraint is invalid]"
        }}
        ...
    ],
}}

- Statements that simply define exact values of parameters are not constraints (e.g., "The cost of producing an X is Y", or "Each X has a size of Y").
- Statements that define bounds are constraints (e.g., "The cost of producing an X is at most Y", or "Each X has a size of at least Y").
- Preferences are not constraints.
- If the same constraint is mentioned multiple times (even in different words), mark all but one of them as "redundant".

Only generate the json file and do not generate anything else. Take a deep breath and approach this task methodically, step by step.

""",
]


def extract_targets_for_lpwp_instance(folder_dir: str, client):
    """Extract targets for the LPWP instance in the given folder.

    Args:
        folder_dir (str): the folder that contains the LPWP instance
    """

    with open(folder_dir + "/input.json", "r") as f:
        input_json = json.load(f)

    description = input_json["description"]

    # Main extraction:
    prompt = prompt_templates[0].format(description=description)

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

    output = output.replace(" \\param", " \\\\\\\\param")

    output_json = output

    if "```json" in output_json:
        # delete until the first '```json'
        output_json = output_json[output_json.find("```json") + 7 :]

        # delete until the last '```'
        output_json = output_json[: output_json.rfind("```")]

    output_json = output_json.replace("```", "")

    update = json.loads(output_json)

    # Sanity check
    prompt = prompt_templates[1].format(
        description=description,
        constraints_json=json.dumps(update["constraints"], indent=4),
    )

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

    output = output.replace(" \\\\param", " \\\\\\\\param")
    output = output.replace(" \\param", " \\\\\\\\param")

    print(output)
    if "```json" in output:
        # delete until the first '```json'
        output = output[output.find("```json") + 7 :]

        # delete until the last '```'
        output = output[: output.rfind("```")]

    checked_update = json.loads(output)

    update["constraints"] = [
        x["definition"] for x in checked_update["constraints"] if x["status"] == "valid"
    ]
    update["description"] = description
    update["parameters"] = input_json["parameters"]

    return update


if __name__ == "__main__":
    client = get_openai_client()
    # for index in range(1, 288):
    #     print(index)

    #     if os.path.exists(f"data/nl4opt/LPWP/prob_{index}/input_targets.json"):
    #         continue

    #     try:
    #         prob_json = extract_targets_for_lpwp_instance(
    #             f"data/nl4opt/LPWP/prob_{index}", client=client
    #         )

    #         with open(f"data/nl4opt/LPWP/prob_{index}/input_targets.json", "w") as f:
    #             json.dump(prob_json, f, indent=4)
    #     except Exception as e:
    #         print(e)
    #         continue

    parent_path = "data/ComplexOR"
    for folder in os.listdir(parent_path):
        if not os.path.isdir(f"{parent_path}/{folder}"):
            continue
        print(folder)
        if os.path.exists(f"{parent_path}/{folder}/input_targets.json"):
            continue

        input_json = json.load(
            open(f"{parent_path}/{folder}/input.json", "r", encoding="utf-8")
        )

        prob_json = extract_targets_for_lpwp_instance(
            f"{parent_path}/{folder}", client=client
        )

        with open(f"{parent_path}/{folder}/input_targets.json", "w") as f:
            json.dump(prob_json, f, indent=4)

    # parent_path = "../data/nlp4lp"

    # for folder in os.listdir(parent_path):
    #     if not os.path.isdir(f"{parent_path}/{folder}"):
    #         continue
    #     print(folder)
    #     if os.path.exists(f"{parent_path}/{folder}/input_targets.json"):
    #         continue

    #     input_json = json.load(
    #         open(f"{parent_path}/{folder}/input.json", "r", encoding="utf-8")
    #     )

    #     prob_json = extract_targets_for_lpwp_instance(
    #         f"{parent_path}/{folder}", client=client
    #     )

    #     with open(f"{parent_path}/{folder}/input_targets.json", "w") as f:
    #         json.dump(prob_json, f, indent=4)
