import json


def read_json(fname):
    return json.load(open(fname, "r"))


def write_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
