import csv
import random

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


def split_data(r_path, w_path):
    content_list = []
    with open(r_path) as r:
        reader = csv.reader(r)
        for index, row in enumerate(reader):
            if index < 1000000:
                if index == 0 or float(row[1]) > 1:
                    content_list.append(row)
            else:
                break
    write_csv(w_path, content_list)


def normalize(input):
    # Normalize the input tensor to [-1, 1]
    # Note: input must be torch.Tensor
    assert isinstance(input, torch.Tensor)
    max_value = torch.max(input.abs()).item()
    return input / max_value


def n_cross_validation(complete_data, n):
    # Note: complete_data must be torch.Tensor
    assert isinstance(complete_data, torch.Tensor)
    assert n > 1, 'n in n_cross_validation must upper than 1'

    divided_data = []
    for i in range(n):
        divided_data.append([])

    for line in complete_data:
        randint = random.randint(0, n - 1)
        divided_data[randint].append(line)

    for i in range(n):
        train_data = []
        tmp_train_data = divided_data[: i] + divided_data[i + 1: ]
        for block in tmp_train_data:
            for line in block:
                train_data.append(line)
        return torch.stack(train_data), torch.stack(divided_data[i])



class NetContainer:

    def __init__(self, size):
        self.size = size

if __name__ == "__main__":
    split_data('file/train.csv', 'set100w.csv')
