import json


def load_configuration(filename):
    with open(f'./configuration/{filename}') as f:
        configuration = json.load(f)
    return configuration
