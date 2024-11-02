import os
import json
from groq import Groq
import openai

groq_key = "###"
openai_key = "###"
openai_org = "###"

groq_client = Groq(api_key=groq_key)
open_ai_client = openai.Client(api_key=openai_key, organization=openai_org)


def extract_json_from_end(text):
    
    try:
        return extract_json_from_end_backup(text)
    except:
        pass
    
    # Find the start of the JSON object
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in the text.")

    # Extract text starting from the first '{'
    json_text = text[json_start:]
    
    # Remove backslashes used for escaping in LaTeX or other formats
    json_text = json_text.replace("\\", "")

    # Remove any extraneous text after the JSON end
    ind = len(json_text) - 1
    while json_text[ind] != "}":
        ind -= 1
    json_text = json_text[: ind + 1]

    # Find the opening curly brace that matches the closing brace
    ind -= 1
    cnt = 1
    while cnt > 0 and ind >= 0:
        if json_text[ind] == "}":
            cnt += 1
        elif json_text[ind] == "{":
            cnt -= 1
        ind -= 1

    # Extract the JSON portion and load it
    json_text = json_text[ind + 1:]

    # Attempt to load JSON
    try:
        jj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

    return jj

def extract_json_from_end_backup(text):

    if "```json" in text:
        text = text.split("```json")[1]
        text = text.split("```")[0]
    ind = len(text) - 1
    while text[ind] != "}":
        ind -= 1
    text = text[: ind + 1]

    ind -= 1
    cnt = 1
    while cnt > 0:
        if text[ind] == "}":
            cnt += 1
        elif text[ind] == "{":
            cnt -= 1
        ind -= 1

    # find comments in the json string (texts between "//" and "\n") and remove them
    while True:
        ind_comment = text.find("//")
        if ind_comment == -1:
            break
        ind_end = text.find("\n", ind_comment)
        text = text[:ind_comment] + text[ind_end + 1 :]

    # convert to json format
    jj = json.loads(text[ind + 1 :])
    return jj


def extract_list_from_end(text):
    ind = len(text) - 1
    while text[ind] != "]":
        ind -= 1
    text = text[: ind + 1]

    ind -= 1
    cnt = 1
    while cnt > 0:
        if text[ind] == "]":
            cnt += 1
        elif text[ind] == "[":
            cnt -= 1
        ind -= 1

    # convert to json format
    jj = json.loads(text[ind + 1 :])
    return jj


# "llama3-70b-8192"
def get_response(prompt, model="llama3-70b-8192"):
    if model == "llama3-70b-8192":
        client = groq_client
    else:
        client = open_ai_client
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    res = chat_completion.choices[0].message.content
    return res


def load_state(state_file):
    with open(state_file, "r") as f:
        state = json.load(f)
    return state


def save_state(state, dir):
    with open(dir, "w") as f:
        json.dump(state, f, indent=4)


def shape_string_to_list(shape_string):
    if type(shape_string) == list:
        return shape_string
    # convert a string like "[N, M, K, 19]" to a list like ['N', 'M', 'K', 19]
    shape_string = shape_string.strip()
    shape_string = shape_string[1:-1]
    shape_list = shape_string.split(",")
    shape_list = [x.strip() for x in shape_list]
    shape_list = [int(x) if x.isdigit() else x for x in shape_list]
    if len(shape_list) == 1 and shape_list[0] == "":
        shape_list = []
    return shape_list


def extract_equal_sign_closed(text):
    ind_1 = text.find("=====")
    ind_2 = text.find("=====", ind_1 + 1)
    obj = text[ind_1 + 6 : ind_2].strip()
    return obj


class Logger:
    def __init__(self, file):
        self.file = file

    def log(self, text):
        with open(self.file, "a") as f:
            f.write(text + "\n")

    def reset(self):
        with open(self.file, "w") as f:
            f.write("")


def create_state(parent_dir, run_dir):
    # read params.json
    with open(os.path.join(parent_dir, "params.json"), "r") as f:
        params = json.load(f)

    data = {}
    for key in params:
        data[key] = params[key]["value"]
        del params[key]["value"]

    # save the data file in the run_dir
    with open(os.path.join(run_dir, "data.json"), "w") as f:
        json.dump(data, f, indent=4)

    # read the description
    with open(os.path.join(parent_dir, "desc.txt"), "r") as f:
        desc = f.read()

    state = {"description": desc, "parameters": params}
    return state

def get_labels(dir):
    with open(os.path.join(dir, "labels.json"), "r") as f:
        labels = json.load(f)
    return labels


if __name__ == "__main__":
    
    text = 'To maximize the number of successfully transmitted shows, we can introduce a new variable called "TotalTransmittedShows". This variable represents the total number of shows that are successfully transmitted.\n\nThe constraint can be formulated as follows:\n\n\\[\n\\text{{Maximize }} TotalTransmittedShows\n\\]\n\nTo model this constraint in the MILP formulation, we need to add the following to the variables list:\n\n\\{\n    "TotalTransmittedShows": \\{\n        "shape": [],\n        "type": "integer",\n        "definition": "The total number of shows transmitted"\n    \\}\n\\}\n\nAnd the following auxiliary constraint:\n\n\\[\n\\forall i \\in \\text{{NumberOfShows}}, \\sum_{j=1}^{\\text{{NumberOfStations}}} \\text{{Transmitted}}[i][j] = \\text{{TotalTransmittedShows}}\n\\]\n\nThe complete output in the requested JSON format is:\n\n\\{\n    "FORMULATION": "",\n    "NEW VARIABLES": \\{\n        "TotalTransmittedShows": \\{\n            "shape": [],\n            "type": "integer",\n            "definition": "The total number of shows transmitted"\n        \\}\n    \\},\n    "AUXILIARY CONSTRAINTS": [\n        ""\n    ]\n\\'
    
    extract_json_from_end(text)