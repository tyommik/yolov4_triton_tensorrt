import sys


def parse_cfg(cfg_file: str):
    """
    Takes a configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    with open(cfg_file, 'r') as cfg:
        blocks = []
        block = {}
        for line in cfg:
            line = line.strip()
            if line.startswith(('#', '\n')) or len(line) <= 0:
                continue
            if line.startswith("["):
                if block:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.rsplit('=')
                value = value.strip()
                key = key.strip().replace(' ', '')
                if value.isnumeric():
                    value = int(value)
                elif not value.isalpha():
                    value = list(map(float, value.split(',')))
                    if len(value) == 1:
                        value = value[0]

                block[key] = value
        blocks.append(block)
        return {i['type']: i for i in blocks}


if __name__ == '__main__':
    config_file = sys.argv[1] + ".cfg"
    blocks = parse_cfg(config_file)
    if not blocks or blocks.get('net') is None:
        raise ValueError("Incorrect config")
    height = blocks['net']['height']
    width = blocks['net']['width']
    classes = blocks['yolo']['classes']
    print(f"{width} {height} {classes}")



