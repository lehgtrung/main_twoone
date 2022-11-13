
import subprocess
import json
import ast
import os
import networkx as nx
from tqdm import tqdm
import numpy as np


atomed_output_path = 'asp_v2/v5/atomed_preds/{iter_number}/{model_number}/{sent_number}.txt'
answerset_output_path = 'asp_v2/v5/answersets/{iter_number}/{model_number}/{sent_number}.txt'
selected_path = 'asp_v2/v5/selected_answersets/{iter_number}/{combined_model_numbers}/{sent_number}.txt'
command = 'clingo --opt-mode=optN asp_v2/v5/p5.lp ' + atomed_output_path + \
          ' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'


def solve(model_number, iter_number, sent_number):
    # Write the program to a file
    process = subprocess.Popen(command.format(model_number=model_number,
                                              iter_number=iter_number,
                                              sent_number=sent_number).split(),
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = [e.split() for e in output.decode().split('\n')[:-2]]
    return result


def convert_to_atoms(pred, prefix):
    atoms = []
    tokens = pred['tokens']
    entities = pred['entities']
    relations = pred['relations']
    for ent in entities:
        c = '{}({}("{}"),1).'.format(prefix, ent[2].lower(), "+".join(tokens[ent[0]:ent[1]]))
        atoms.append(c)
    for rel in relations:
        c = '{}({}("{}","{}"),1).'.format(prefix,
                                            ''.join([rel[4].split('_')[0].lower(), *rel[4].split('_')[1:]]),
                                            '+'.join(tokens[rel[0]:rel[1]]),
                                            '+'.join(tokens[rel[2]:rel[3]]))
        atoms.append(c)
    return atoms


def write_down_a_list(path, lst):
    with open(path, 'w') as f:
        f.writelines(map(lambda x: x + '\n', lst))


def convert_to_consistent_answersets(preds_path, iter_number, model_number):
    os.makedirs(os.path.dirname(atomed_output_path.format(
            iter_number=iter_number,
            model_number=model_number,
            sent_number=0)),
        exist_ok=True)
    os.makedirs(os.path.dirname(answerset_output_path.format(
            iter_number=iter_number,
            model_number=model_number,
            sent_number=0)),
        exist_ok=True)
    # Load the predictions
    with open(preds_path, 'r') as f:
        preds = json.load(f)
    # Convert them into atomed form
    for i, row in enumerate(preds):
        atoms = convert_to_atoms(row, prefix='atom')
        # Path to write the atomed preds down
        path = atomed_output_path.format(iter_number=iter_number,
                                         model_number=model_number,
                                         sent_number=i)
        write_down_a_list(path, atoms)
    # Convert the atomed preds to answersets
    for i in tqdm(range(len(preds))):
        path = answerset_output_path.format(iter_number=iter_number,
                                            model_number=model_number,
                                            sent_number=i)
        answersets = solve(model_number=model_number,
                           iter_number=iter_number,
                           sent_number=i)
        with open(path, 'w') as f:
            for answerset in answersets:
                f.writelines(map(lambda x: x+'\n', answerset))
                f.write('BREAK\n')


def split_at_values(lst, value):
    indices = [i + 1 for i, x in enumerate(lst) if x == value]
    split_lst = [lst[i:j] for i, j in zip([0] + indices, indices + [None])]
    return [e[:-1] for e in split_lst if e != []]


def parse_answersets_from_file(path, with_break):
    with open(path, 'r') as f:
        answersets = [e.strip('\n').replace(' ', '') for e in f.readlines()]
    if with_break:
        answersets = split_at_values(answersets, 'BREAK')
    return list(answersets)


def compute_set_diff(answerset1, answerset2):
    n = len(answerset1)
    m = len(answerset2)
    intersect = set(answerset1).intersection(answerset2)
    return n + m - 2*len(intersect)


def create_dist_graph(all_answersets, self=False):
    graph = nx.Graph()

    for i, answersets in enumerate(all_answersets):
        for j, other_answersets in enumerate(all_answersets):
            if j > i:
                for k, answerset in enumerate(answersets):
                    for g, other_answerset in enumerate(other_answersets):
                        weight = compute_set_diff(answerset, other_answerset)
                        graph.add_edge(f'{i}.{k}', f'{j}.{g}', weight=weight)
    if self:
        for i, answersets in enumerate(all_answersets):
            for j in range(len(answersets)):
                for k in range(len(answersets)):
                    if k > j:
                        weight = compute_set_diff(answersets[i], answersets[j])
                        graph.add_edge(f'{i}.{j}', f'{i}.{k}', weight=weight)
    return graph


def find_center_vertex(sent_number, iter_number, model_numbers):
    all_answersets = []
    for model_number in model_numbers:
        path = answerset_output_path.format(iter_number=iter_number,
                                            model_number=model_number,
                                            sent_number=sent_number)
        answersets = parse_answersets_from_file(path, with_break=True)
        all_answersets.append(answersets)
    graph = create_dist_graph(all_answersets)
    centrality = nx.closeness_centrality(graph)
    center = max(centrality, key=centrality.get)
    part, index = center.split('.')
    part = int(part)
    index = int(index)
    return all_answersets[part][index]


def select_answerset(iter_number, model_numbers):
    model_numbers = sorted(list(model_numbers))
    combined_model_numbers = '.'.join(map(str, model_numbers))
    os.makedirs(os.path.dirname(selected_path.format(iter_number=iter_number,
                                                     combined_model_numbers=combined_model_numbers,
                                                     sent_number=0)),
                exist_ok=True)
    n = len(os.listdir(os.path.dirname(answerset_output_path.format(iter_number=iter_number,
                                                                    model_number=model_numbers[0],
                                                                    sent_number=0))))
    # For each sentence find the answerset that minimize avg set diff
    for i in tqdm(range(n)):
        path = selected_path.format(iter_number=iter_number,
                                    combined_model_numbers=combined_model_numbers,
                                    sent_number=i)
        selected_answerset = find_center_vertex(sent_number=i,
                                                iter_number=iter_number,
                                                model_numbers=model_numbers)
        write_down_a_list(path, selected_answerset)


def compare_with_gt(gt_path, iter_number, model_numbers):
    model_numbers = sorted(list(model_numbers))
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    all_gt_atoms = []
    for row in gt:
        gt_atoms = convert_to_atoms(row, prefix='ok')
        all_gt_atoms.append(gt_atoms)
    total_set_diff = []
    for i in range(len(all_gt_atoms)):
        combined_model_numbers = '.'.join(map(str, model_numbers))
        path = selected_path.format(iter_number=iter_number,
                                    combined_model_numbers=combined_model_numbers,
                                    sent_number=i)
        answerset = parse_answersets_from_file(path, with_break=False)
        set_diff = compute_set_diff(all_gt_atoms[i], answerset)
        total_set_diff.append(set_diff)
    print('Number of hard match: ', len([e for e in total_set_diff if e == 0]))
    print('Average set diff: ', np.mean(total_set_diff))


def compare_raw_with_gt(raw_path, gt_path):
    with open(raw_path, 'r') as f:
        raw = json.load(f)
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    all_gt_atoms = []
    for row in gt:
        gt_atoms = convert_to_atoms(row, prefix='ok')
        all_gt_atoms.append(gt_atoms)
    all_raw_atoms = []
    for row in raw:
        raw_atoms = convert_to_atoms(row, prefix='ok')
        all_raw_atoms.append(raw_atoms)
    total_set_diff = []
    for i, (gt_atoms, raw_atoms) in enumerate(zip(all_gt_atoms, all_raw_atoms)):
        set_diff = compute_set_diff(gt_atoms, raw_atoms)
        total_set_diff.append(set_diff)
    print('Number of hard match: ', len([e for e in total_set_diff if e == 0]))
    print('Average set diff: ', np.mean(total_set_diff))


if __name__ == '__main__':
    _iter = 1
    for _model_number in range(3):
        convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                         iter_number=_iter,
                                         model_number=_model_number)
    select_answerset(_iter, [0, 1])
    select_answerset(_iter, [0, 2])
    select_answerset(_iter, [1, 2])
    select_answerset(_iter, [0, 1, 2])

    print('Compare joint M0,M1 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    _iter, [0, 1])
    print('===================================')
    print('Compare joint M0,M2 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    _iter, [0, 2])
    print('===================================')
    print('Compare joint M1,M2 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    _iter, [1, 2])
    print('===================================')
    print('Compare joint M0,M1,M2 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    _iter, [0, 1, 2])
    print('***********************************')
    print('Compare M0 with gt')
    compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_0.json',
                        gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    print('===================================')
    print('Compare M1 with gt')
    compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_1.json',
                        gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    print('===================================')
    print('Compare M2 with gt')
    compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_2.json',
                        gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')

