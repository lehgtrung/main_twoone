
import subprocess
import json
import ast
import os
import networkx as nx
from tqdm import tqdm
import numpy as np
import glob
import random
import re
from independent_evaluation import evaluate_model


atomed_output_path = 'asp_v2/v5/atomed_preds/{iter_number}/{model_number}/{sent_number}.txt'
answerset_output_path = 'asp_v2/v5/answersets/{iter_number}/{model_number}/{sent_number}.txt'
selected_path = 'asp_v2/v5/selected_answersets/{iter_number}/{combined_model_numbers}/{sent_number}.txt'
command = 'clingo --opt-mode=optN asp_v2/v5/p5_with_rules.lp ' + atomed_output_path + \
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
        c = '{}({}("{}"),1).'.format(prefix, ent[2].lower(), '{}+{}'.format(ent[0], ent[1]))
        atoms.append(c)
    for rel in relations:
        c = '{}({}("{}","{}"),1).'.format(prefix,
                                          ''.join([rel[4].split('_')[0].lower(), *rel[4].split('_')[1:]]),
                                          '{}+{}'.format(rel[0], rel[1]),
                                          '{}+{}'.format(rel[2], rel[3]))
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
                        weight = compute_set_diff(answersets[j], answersets[k])
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
    print('Number of hard matches: ', len([e for e in total_set_diff if e == 0]))
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
    print('Number of hard matches: ', len([e for e in total_set_diff if e == 0]))
    print('Average set diff: ', np.mean(total_set_diff))


def convert_atom_to_tuple_form(atom):
    atom = atom.lstrip('atom').lstrip('(').lstrip('ok').lstrip('(').rstrip(').').rstrip(',1').strip()
    typ = 'relation'
    mapper = {
        'locatedIn': 'Located_In',
        'kill': 'Kill',
        'orgbasedIn': 'OrgBased_In',
        'liveIn': 'Live_In',
        'workFor': 'Work_For'
    }
    if atom.split('(')[0] in ['loc', 'peop', 'org', 'other']:
        typ = 'entity'
    entity_pattern = re.compile(r'(\w+)\("([0-9]+\+[0-9]+)"\)')
    relation_pattern = re.compile(r'(\w+)\("([0-9]+\+[0-9]+)","([0-9]+\+[0-9]+)"\)')
    if typ == 'entity':
        ret = re.findall(entity_pattern, atom)[0]
        name = ret[0]
        start, end = ret[1].split('+')
        return typ, (int(start), int(end), name.capitalize())
    else:
        ret = re.findall(relation_pattern, atom)[0]
        name = ret[0]
        head_start, head_end = ret[1].split('+')
        tail_start, tail_end = ret[2].split('+')
        return typ, (int(head_start), int(head_end), int(tail_start), int(tail_end), mapper[name])


def convert_atoms_to_file_form(tokens_list, atoms_list):
    data = []
    for tokens, atoms in zip(tokens_list, atoms_list):
        entities = []
        relations = []
        for atom in atoms:
            typ, obj = convert_atom_to_tuple_form(atom)
            if typ == 'entity':
                entities.append(obj)
            else:
                relations.append(obj)
        data.append({
            'tokens': tokens,
            'entities': entities,
            'relations': relations
        })
    return data


def compare_raw_selection_with_gt(raw_path1, raw_path2, gt_path):
    with open(raw_path1, 'r') as f:
        raw1 = json.load(f)
    with open(raw_path2, 'r') as f:
        raw2 = json.load(f)
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    all_gt_atoms = []
    for row in gt:
        gt_atoms = convert_to_atoms(row, prefix='ok')
        all_gt_atoms.append(gt_atoms)
    selected_raw_indices = []
    selected_answersets = []
    tokens_list = []
    for i, (row1, row2) in enumerate(zip(raw1, raw2)):
        raw_atoms1 = convert_to_atoms(row1, prefix='ok')
        raw_atoms2 = convert_to_atoms(row2, prefix='ok')
        set_diff = compute_set_diff(raw_atoms1, raw_atoms2)
        if set_diff == 0:
            tokens_list.append(raw1[i]['tokens'])
            selected_raw_indices.append(i)
            selected_answersets.append(raw_atoms1)

    num_agreements = len(selected_raw_indices)
    print('Number of agreements: ', num_agreements)
    print('Percentage of agreements: ', num_agreements/len(gt))

    # Convert selected sentences
    gts = [gt[i] for i in selected_raw_indices]
    preds = convert_atoms_to_file_form(tokens_list, selected_answersets)
    evaluate_model(preds, gts)


def compare_asp_selection_with_gt(iter_number,
                                  model_number1,
                                  model_number2,
                                  gt_path):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    all_gt_atoms = []
    for row in gt:
        gt_atoms = convert_to_atoms(row, prefix='ok')
        all_gt_atoms.append(gt_atoms)
    meta_paths1 = glob.glob(answerset_output_path.format(iter_number=iter_number,
                                                         model_number=model_number1,
                                                         sent_number='*'))
    meta_paths2 = glob.glob(answerset_output_path.format(iter_number=iter_number,
                                                         model_number=model_number2,
                                                         sent_number='*'))
    meta_paths1 = sorted(meta_paths1)
    meta_paths2 = sorted(meta_paths2)
    assert len(meta_paths1) == len(meta_paths2)
    selected_indices = []
    selected_answersets = []
    tokens_list = []
    for path1, path2 in zip(meta_paths1, meta_paths2):
        i = int(os.path.basename(path1).split('.')[0])
        all_answersets1 = parse_answersets_from_file(path1, with_break=True)
        all_answersets2 = parse_answersets_from_file(path2, with_break=True)
        set_all_answersets1 = list(map(tuple, all_answersets1))
        set_all_answersets2 = list(map(tuple, all_answersets2))
        intersection = list(set(set_all_answersets1).intersection(set_all_answersets2))
        if len(intersection) > 0:
            # if len(intersection) > 1:
            #     print('MORE THAN 1 AGREEMENTS')
            tokens_list.append(gt[i]['tokens'])
            selected_indices.append(i)
            selected_answersets.append(intersection[random.choice(range(len(intersection)))])

    num_agreements = len(selected_indices)
    print('Number of agreements: ', num_agreements)
    print('Percentage of agreements: ', num_agreements/len(gt))

    # Convert selected sentences
    gts = [gt[i] for i in selected_indices]
    preds = convert_atoms_to_file_form(tokens_list, selected_answersets)
    evaluate_model(preds, gts)


def compare_selection_with_gt(gt_path,
                              pred_path1,
                              pred_path2,
                              pred_path3,
                              iter_number):
    with open(pred_path1, 'r') as f:
        pred1 = json.load(f)
    with open(pred_path2, 'r') as f:
        pred2 = json.load(f)
    with open(pred_path3, 'r') as f:
        pred3 = json.load(f)


    # Evaluate each of them
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    print('Model 1')
    evaluate_model(pred1, gt)
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    print('Model 2')
    evaluate_model(pred2, gt)
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    print('Model 3')
    evaluate_model(pred3, gt)

    # aggregate 3 models
    answerset_list = []
    tokens_list = []

    for i in range(len(gt)):
        combined_model_numbers = '.'.join(map(str, range(3)))
        path = selected_path.format(iter_number=iter_number,
                                    combined_model_numbers=combined_model_numbers,
                                    sent_number=i)
        answerset = parse_answersets_from_file(path, with_break=False)
        answerset_list.append(answerset)
        tokens_list.append(gt[i]['tokens'])
    preds = convert_atoms_to_file_form(tokens_list, answerset_list)

    with open(gt_path, 'r') as f:
        gt = json.load(f)
    print('Model AGG')
    evaluate_model(preds, gt)


def does_every_entity_has_relation(gt_path):
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    for row in gt:
        tokens = row['tokens']
        entities = row['entities']
        relations = row['relations']
        for ent in entities:
            start, end, etype = ent
            if etype != 'Other':
                flag = False
                for rel in relations:
                    head_start, head_end, tail_start, tail_end, rtype = rel
                    if (start, end) == (head_start, head_end) or (start, end) == (tail_start, tail_end):
                        flag = True
                if not flag:
                    print(tokens)
                    print(f'{etype}({" ".join(tokens[start:end])})')
                    print('=================')


def convert_json_to_asp_form(json_path, prefix='ok'):
    with open(json_path, 'r') as f:
        rows = json.load(f)
    all_atoms = []
    for row in rows:
        atoms = convert_to_atoms(row, prefix=prefix)
        all_atoms.append(atoms)
    return all_atoms


def compare_triple_sets(control, treatment1, treatment2):
    intersect = set.intersection(set(control), set(treatment1), set(treatment2))
    unique2set1 = set(treatment1) - set(control)
    unique2set2 = set(treatment2) - set(control)
    return intersect, unique2set1, unique2set2


def how_many_sentences_are_modified(iter_number, model_number):
    raw_path = f'asp_v2/v5/preds/iter={iter_number}/prediction_{model_number}.json'
    gt_path = 'datasets/core_conll04/conll04_30/fold=1/unlabeled.json'
    all_raw_atoms = convert_json_to_asp_form(raw_path)
    all_gt_atoms = convert_json_to_asp_form(gt_path)

    mod_count = 0
    for i in range(len(all_raw_atoms)):
        path = answerset_output_path.format(iter_number=iter_number,
                                            model_number=model_number,
                                            sent_number=i)
        answersets = parse_answersets_from_file(path, with_break=True)
        if len(answersets) > 1:
            mod_count += 1
            continue
        set_diff = compute_set_diff(all_raw_atoms[i], answersets[0])
        if set_diff > 0:
            # print('ASP: ', answersets[0])
            # print('RAW: ', all_raw_atoms[i])
            # print('GT: ', all_gt_atoms[i])
            # diff_asp_raw = compute_set_diff(all_raw_atoms[i], answersets[0])
            # diff_asp_gt = compute_set_diff(all_gt_atoms[i], answersets[0])
            # diff_gt_raw = compute_set_diff(all_gt_atoms[i], all_raw_atoms[i])
            # print('Diff(asp, raw) = ', diff_asp_raw)
            # print('Diff(asp, gt) = ', diff_asp_gt)
            # print('Diff(gt, raw) = ', diff_gt_raw)
            # intersect, unique2asp, unique2raw = compare_triple_sets(all_gt_atoms[i],
            #                                                         answersets[0],
            #                                                         all_raw_atoms[i])
            # print('Common atoms: ', intersect)
            # print('Unique 2 asp: ', unique2asp)
            # print('Unique 2 raw: ', unique2raw)
            # if diff_asp_gt < diff_gt_raw:
            #     print('ASP wins')
            # elif diff_asp_gt > diff_gt_raw:
            #     print('Raw win')
            # else:
            #     print('Draw')
            # print('======================')
            # input()
            mod_count += 1
    print(mod_count)


if __name__ == '__main__':
    _iter = 3
    for _model_number in range(3):
        convert_to_consistent_answersets(f'asp_v2/v5/preds/iter={_iter}/prediction_{_model_number}.json',
                                         iter_number=_iter,
                                         model_number=_model_number)
    # select_answerset(_iter, [0, 1])
    # select_answerset(_iter, [0, 2])
    # select_answerset(_iter, [1, 2])
    select_answerset(_iter, [0, 1, 2])

    compare_selection_with_gt(gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
                              pred_path1=f'asp_v2/v5/preds/iter={_iter}/prediction_0.json',
                              pred_path2=f'asp_v2/v5/preds/iter={_iter}/prediction_1.json',
                              pred_path3=f'asp_v2/v5/preds/iter={_iter}/prediction_2.json',
                              iter_number=_iter)

    #
    # print('Compare joint M0,M1 with gt')
    # compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
    #                 _iter, [0, 1])
    # print('===================================')
    # print('Compare joint M0,M2 with gt')
    # compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
    #                 _iter, [0, 2])
    # print('===================================')
    # print('Compare joint M1,M2 with gt')
    # compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
    #                 _iter, [1, 2])
    # print('===================================')
    # print('Compare joint M0,M1,M2 with gt')
    # compare_with_gt('datasets/core_conll04/conll04_30/fold=1/unlabeled.json',
    #                 _iter, [0, 1, 2])
    # print('***********************************')
    # print('Compare M0 with gt')
    # compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_0.json',
    #                     gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('===================================')
    # print('Compare M1 with gt')
    # compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_1.json',
    #                     gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('===================================')
    # print('Compare M2 with gt')
    # compare_raw_with_gt(raw_path=f'asp_v2/v5/preds/iter={_iter}/prediction_2.json',
    #                     gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')

    # model_number1 = 0
    # model_number2 = 1
    # print(f'MODEL NUMBER = {model_number1},{model_number2}')
    # print('RAW SELECTION')
    # compare_raw_selection_with_gt(raw_path1=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number1}.json',
    #                               raw_path2=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number2}.json',
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('++++++++++++++++++++++++++++++++++')
    # print('ASP SELECTION')
    # compare_asp_selection_with_gt(iter_number=_iter,
    #                               model_number1=model_number1,
    #                               model_number2=model_number2,
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    #
    # print('**********************************************')
    # model_number1 = 0
    # model_number2 = 2
    # print(f'MODEL NUMBER = {model_number1},{model_number2}')
    # print('RAW SELECTION')
    # compare_raw_selection_with_gt(raw_path1=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number1}.json',
    #                               raw_path2=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number2}.json',
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('++++++++++++++++++++++++++++++++++')
    # print('ASP SELECTION')
    # compare_asp_selection_with_gt(iter_number=_iter,
    #                               model_number1=model_number1,
    #                               model_number2=model_number2,
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('**********************************************')
    # model_number1 = 1
    # model_number2 = 2
    # print(f'MODEL NUMBER = {model_number1},{model_number2}')
    # print('RAW SELECTION')
    # compare_raw_selection_with_gt(raw_path1=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number1}.json',
    #                               raw_path2=f'asp_v2/v5/preds/iter={_iter}/prediction_{model_number2}.json',
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # print('++++++++++++++++++++++++++++++++++')
    # print('ASP SELECTION')
    # compare_asp_selection_with_gt(iter_number=_iter,
    #                               model_number1=model_number1,
    #                               model_number2=model_number2,
    #                               gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')

    # does_every_entity_has_relation(gt_path='datasets/core_conll04/conll04_30/fold=1/unlabeled.json')
    # _iter = 0
    # _model_number = 0
    # print('Model=0')
    # how_many_sentences_are_modified(_iter, _model_number)
    # _model_number = 1
    # print('Model=1')
    # how_many_sentences_are_modified(_iter, _model_number)
    # _model_number = 2
    # print('Model=2')
    # how_many_sentences_are_modified(_iter, _model_number)