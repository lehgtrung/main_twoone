import json
import random
import os
from tqdm import tqdm


def split_data_for_ssl(in_path, out_path, portion):
    with open(in_path, 'r') as f:
        data = json.load(f)
    n = len(data)
    m = int(n*portion)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[:m]
    labeled = []
    unlabeled = []
    for i, row in enumerate(data):
        if i in indices:
            labeled.append(row)
        else:
            unlabeled.append(row)
    with open(out_path.format('labeled.json'), 'w') as f:
        json.dump(labeled, f)
    with open(out_path.format('unlabeled.json'), 'w') as f:
        json.dump(unlabeled, f)
    with open(out_path.format('indices.json'), 'w') as f:
        json.dump(indices, f)


def gen_data_folds(in_path):
    # Split the data into 30-70
    base_path = './datasets/conll04/folds_5/{}'
    for i in tqdm(range(10)):
        path = base_path.format(i+1)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, '{}')
        split_data_for_ssl(in_path, out_path=path, portion=0.05)


if __name__ == '__main__':
    gen_data_folds('./datasets/unified/train.CoNLL04.json')