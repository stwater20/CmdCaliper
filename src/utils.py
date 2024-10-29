import json
import yaml

def load_yaml(path_to_data):
    with open(path_to_data, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_json(path_to_data):
    with open(path_to_data, "r") as f:
        data = json.load(f)
    return data

def save_json(data, path_to_data):
    with open(path_to_data, "w") as f:
        json.dump(data, f, indent=2)

def postprocess_data_synthesis_response(response):
    if "```" in response:
        response_list = [r for r in response.split("\n") if "```" not in r]
        response = "\n".join(response_list)
    return response

def extract_cmds(response):
    response = response.replace("<CMD>\n", "<CMD>")

    new_generated_cmd_list = []
    for r in response.split("\n"):
        r = r.strip()
        if "<CMD>" in r:
            cmd = r[len("<CMD>"):]
            if "</CMD>" in r:
                cmd = cmd[:cmd.index("</CMD>")]
            if cmd == "":
                continue
            new_generated_cmd_list.append(cmd.strip())
    return new_generated_cmd_list
