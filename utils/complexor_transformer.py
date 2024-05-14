import openai
import json
import os
import sys

from misc import get_openai_client


prompt_template = """
You are an expert mathematical modeler and an optimization professor at a top university.
Your task is to first read a given optimization problem and a list of related parameters. You should then go through the statements one by one, and re-write the description so it contains all of the parameters and explains them. Moreover, you should generate an updated list of parameters with an added definition for each. Here is an example:

*** input ***

Aircraft Allocation problem involves assign aircrafts to satisfy the demand of different routes. The capacitiy of each aircraft for different routes varies. You should note that the allocation of a aircraft should not exceed its availability. Also, the cost value indicates the cost for an allocation.   

parameters:
[
    {{
      "symbol": "AircraftNum",
      "shape": []
    }},
    {{
      "symbol": "RouteNum",
      "shape": []
    }},
    
    {{
      "symbol": "Availability",
      "shape": [AirCraftNum]
    }},
    {{
        "symbol": "Demand",
        "shape": [RouteNum]
    }},
    {{
        "symbol": "Capabilities",
        "shape": [AirCraftNum, RouteNum]
    }},
    {{
        "symbol": "Cost",
        "shape": [AirCraftNum, RouteNum]
    }}
]

*** output ***

{{
    "description": "Aircraft Allocation problem involves assigning \param{{AircraftNum}} aircrafts to satisfy the demand of \param{{RouteNum}} different routes. Demand of route i is \param{{Demand}}, and the capacitiy of each aircraft i for each route j is \param{{Capabilities_{{i,j}}}}. You should note that the allocation of a aircraft i should not exceed its availability \param{{Availability_i}}. Also, \param{{Cost_{{i,j}}}} indicates the cost of assigning aircraft i to route j.",
    
    parameters:[
        {{
            "symbol": "AircraftNum",
            "definition": "The number of aircrafts",
            "shape": []
        }},
        {{
            "symbol": "RouteNum",
            "definition": "The number of routes",
            "shape": []
        }},
        {{
            "symbol": "Availability",
            "definition": "The availability of each aircraft",
            "shape": [AirCraftNum]
        }},
        {{
            "symbol": "Demand",
            "definition": "The demand of each route",
            "shape": [RouteNum]
        }},
        {{
            "symbol": "Capabilities",
            "definition": "The capabilities of each aircraft for each route",
            "shape": [AirCraftNum, RouteNum]
        }},
        {{
            "symbol": "Cost",
            "definition": "The cost of assigning each aircraft to each route",
            "shape": [AirCraftNum, RouteNum]
        }}
    ]
}}
    

Here is the problem description:

-----
{description}
-----


And here is the list of parameters:

-----
{parameters}
-----


- Only generate the requested json file, and nothing else.
- Use \\param{{}} to indicate the parameters. Note that indices are not parameters (For example use "\\param{{cost_i}} is the cost of the i-th good" instead of "\\param{{cost_i}} is the cost of the \\param{{i}}-th good").
- Remove ` and ``` chracters from the description.
- Generate the full updated json file, including the description and the list of parameters, and do not omit anything.
- Feel free to rename the parameters to give them more meaningful names. Use CamelCase for parameter names.
- Use ```json and ``` to enclose the json file.
Take a deep breath and approach this task methodically, step by step.

"""


def transform_complexor_instance(folder_path: str, client):
    """Clean the LPWP instance in the given folder.

    Args:
        folder_dir (str): the folder that contains the LPWP instance
    """

    with open(folder_path + "parsed.json", "r") as f:
        parsed_file = json.load(f)

    # with open(folder_path + "data.json", "r") as f:
    #     data = json.load(f)

    if "desc" in parsed_file:
        parsed_file["description"] = parsed_file["desc"]
        del parsed_file["desc"]

    data = parsed_file["model"]
    if "param" in data:
        data["parameter"] = data["param"]
        del data["param"]

    description = parsed_file["description"]
    parameters = []

    for item in data["set"]:
        parameters.append({"symbol": item["name"], "shape": []})

    for item in data["parameter"]:
        if "indexDomain" in item:
            item["domain"] = item["indexDomain"]
            del item["indexDomain"]
        if not "domain" in item:
            item["domain"] = []
        shape = item["domain"]
        if type(shape) == str:
            shape = [shape]
        parameters.append({"symbol": item["name"], "shape": shape})

    print(json.dumps(parameters, indent=4))
    prompt = prompt_template.format(
        description=parsed_file["description"],
        parameters=json.dumps(parameters, indent=4),
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

    output = output.replace("\\param", "\\\\param")

    output_json = output.split("```json")[1].split("```")[0]

    with open("tmp.json", "w") as f:
        f.write(output_json)

    update = json.loads(output_json)

    output_desc = update["description"]
    for param in update["parameters"]:
        if "_" in param["symbol"]:
            new_symbol = param["symbol"].replace("_", "")
            output_desc = output_desc.replace(param["symbol"], new_symbol)
            param["symbol"] = new_symbol

    update["description"] = output_desc

    return update


if __name__ == "__main__":
    client = get_openai_client()
    # iterate over all files in data/ComplexOR,
    idx = 0
    for file in os.listdir("data/ComplexOR"):
        # if os.path.isdir(f"data/ComplexOR/{file}"):
        #     continue

        # if not file.endswith("parsed.json"):
        #     continue

        # if not os.path.exists(f"data/ComplexOR/{file[:-11]}data.json"):
        #     continue

        # # create a new folder in data/ComplexOR/file_name
        # os.mkdir(f"data/ComplexOR/{file[:-12]}")
        # # move the file into the new folder
        # os.rename(
        #     f"data/ComplexOR/{file}",
        #     f"data/ComplexOR/{file[:-12]}/parsed.json",
        # )
        # os.rename(
        #     f"data/ComplexOR/{file[:-11]}data.json",
        #     f"data/ComplexOR/{file[:-12]}/data.json",
        # )

        if not os.path.isdir(f"data/ComplexOR/{file}"):
            continue

        if os.path.exists(f"data/ComplexOR/{file}/input.json"):
            continue

        print(f"### Solving {file}...")
        prob_json = transform_complexor_instance(
            f"data/ComplexOR/{file}/", client=client
        )

        with open(f"data/ComplexOR/{file}/input.json", "w") as f:
            json.dump(prob_json, f, indent=4)
