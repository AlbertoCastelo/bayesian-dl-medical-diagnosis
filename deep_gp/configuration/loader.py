import json


def load_configuration(filename, path='./configuration'):
    with open(f'{path}/{filename}') as f:
        configuration = json.load(f)
    return configuration
