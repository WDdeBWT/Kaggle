import csv
import torch
from queue import Queue


def read_csv(r_path):
    content_list = []
    with open(r_path) as r:
        reader = csv.reader(r)
        for row in reader:
            content_list.append(row)
    return content_list


def write_csv(w_path, content_list):
    with open(w_path, 'w', newline="") as w:
        writer = csv.writer(w)
        if content_list:
            for line in content_list:
                if line[0]:
                    line[0] = line[0].strip()
                writer.writerow(line)


def normalize(input):
    # Normalize the input tensor to [-1, 1]
    max_value = torch.max(input.abs()).item()
    return input / max_value


class NetContainer:

    def __init__(self, size):
        self.size = size
