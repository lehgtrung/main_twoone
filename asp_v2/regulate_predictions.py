
import subprocess
import json
import ast
import os
import networkx as nx
from tqdm import tqdm
import numpy as np


atomed_output_path = 'asp_v2/v5/atomed_preds/{model_number}/{sent_number}.txt'
answerset_output_path = 'asp_v2/v5/answersets/{model_number}/{sent_number}.txt'
selected_path = 'asp_v2/v5/selected_answersets/{model_number1}.{model_number2}/{sent_number}.txt'
command = 'clingo --opt-mode=optN asp_v2/v5/p5.lp ' + atomed_output_path + \
          ' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'


def solve(model_number, sent_number):
    # Write the program to a file
    process = subprocess.Popen(command.format(model_number=model_number,
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


def convert_to_consistent_answersets(preds_path, model_number):
    os.makedirs(os.path.dirname(atomed_output_path.format(model_number=model_number,
                                                          sent_number=0)),
                exist_ok=True)
    os.makedirs(os.path.dirname(answerset_output_path.format(model_number=model_number,
                                                             sent_number=0)),
                exist_ok=True)
    # Load the predictions
    with open(preds_path, 'r') as f:
        preds = json.load(f)
    # Convert them into atomed form
    for i, row in enumerate(preds):
        atoms = convert_to_atoms(row, prefix='atom')
        # Path to write the atomed preds down
        path = atomed_output_path.format(model_number=model_number,
                                         sent_number=i)
        write_down_a_list(path, atoms)
    # Convert the atomed preds to answersets
    for i in tqdm(range(len(preds))):
        path = answerset_output_path.format(model_number=model_number,
                                            sent_number=i)
        answersets = solve(model_number=model_number, sent_number=i)
        # with open(path, 'w') as f:
        #     json.dump(answersets, f)
        # if len(answersets) > 1:
        #     print(i)
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


def create_dist_graph(answersets1, answersets2, self=False):
    graph = nx.Graph()
    for i, answerset1 in enumerate(answersets1):
        for j, answerset2 in enumerate(answersets2):
            weight = compute_set_diff(answerset1, answerset2)
            graph.add_edge(f'left.{i}', f'right.{j}', weight=weight)
    if self:
        for i in range(len(answersets1)):
            for j in range(len(answersets1)):
                if i < j:
                    weight = compute_set_diff(answersets1[i], answersets1[j])
                    graph.add_edge(f'left.{i}', f'left.{j}', weight=weight)
        for i in range(len(answersets2)):
            for j in range(len(answersets2)):
                if i < j:
                    weight = compute_set_diff(answersets2[i], answersets2[j])
                    graph.add_edge(f'right.{i}', f'right.{j}', weight=weight)
    return graph


def find_center_vertex(model_number1, model_number2, sent_number):
    # Read 2 files, parse them and create a graph
    path = answerset_output_path.format(model_number=model_number1,
                                        sent_number=sent_number)
    answersets0 = parse_answersets_from_file(path, with_break=True)
    path = answerset_output_path.format(model_number=model_number2,
                                        sent_number=sent_number)
    answersets1 = parse_answersets_from_file(path, with_break=True)
    graph = create_dist_graph(answersets0, answersets1, self=False)
    centrality = nx.closeness_centrality(graph)
    center = max(centrality, key=centrality.get)
    part, index = center.split('.')
    index = int(index)
    if part == 'left':
        return answersets0[index]
    return answersets1[index]


def select_answerset(model_number1, model_number2):
    os.makedirs(os.path.dirname(selected_path.format(model_number1=model_number1,
                                                     model_number2=model_number2,
                                                     sent_number=0)),
                exist_ok=True)
    n = len(os.listdir(os.path.dirname(answerset_output_path.format(model_number=model_number1,
                                                                    sent_number=0))))
    # For each sentence find the answerset that minimize avg set diff
    for i in tqdm(range(n)):
        path = selected_path.format(model_number1=model_number1,
                                    model_number2=model_number2,
                                    sent_number=i)
        selected_answerset = find_center_vertex(model_number1=model_number1,
                                                model_number2=model_number2,
                                                sent_number=i)
        write_down_a_list(path, selected_answerset)


def compare_with_gt(gt_path, model_number1, model_number2):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    all_gt_atoms = []
    for row in gt:
        gt_atoms = convert_to_atoms(row, prefix='ok')
        all_gt_atoms.append(gt_atoms)
    total_set_diff = []
    for i in range(len(all_gt_atoms)):
        path = selected_path.format(model_number1=model_number1,
                                    model_number2=model_number2,
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
    # TODO:
    # Combine 3 models instead of 2 models
    _iter = 1
    _model_number = 0
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    _model_number = 1
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    select_answerset(model_number1=0,
                     model_number2=1)

    _model_number = 0
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    _model_number = 2
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    select_answerset(model_number1=0,
                     model_number2=2)

    _model_number = 1
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    _model_number = 2
    convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                     _model_number)
    select_answerset(model_number1=1,
                     model_number2=2)

    print('Compare joint M0,M1 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    model_number1=0,
                    model_number2=1)
    print('===================================')
    print('Compare joint M0,M2 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    model_number1=0,
                    model_number2=2)
    print('===================================')
    print('Compare joint M1,M2 with gt')
    compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                    model_number1=1,
                    model_number2=2)
    print('===================================')
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

