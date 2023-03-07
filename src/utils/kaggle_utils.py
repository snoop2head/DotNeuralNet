import json


def get_alphabet_map(path="./src/utils/alphabet_map.json"):

    with open(path, "r") as f:
        alphabet_map = json.load(f)

    return alphabet_map
