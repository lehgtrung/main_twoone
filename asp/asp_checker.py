
import json
import random
import os
from .asp_converter import is_satisfiable
from .asp_ult import match_form


def model_exists(path):
    if os.path.exists(os.path.join(os.path.dirname(path), 'labeled.pt')):
        return True
    return False


def unify_two_datasets(labeled_path, pseudo_path, output_path):
    with open(labeled_path, 'r') as f:
        labeled = json.load(f)
        for line in labeled:
            line['eweights'] = [1.0 for _ in range(len(line['entities']))]
            line['rweights'] = [1.0 for _ in range(len(line['relations']))]
    with open(pseudo_path, 'r') as f:
        pseudo = json.load(f)
    with open(output_path, 'w') as f:
        json.dump(labeled + pseudo, f)


def check_convergence(iteration, max_iterations, raw_pseudo_labeled_path, logger):
    with open('asp/satisfiable.lp') as f:
        satisfiable_program = f.read()
    count = 0
    with open(raw_pseudo_labeled_path, 'r') as f:
        data = json.load(f)
        for row in data:
            if not is_satisfiable(row['entities'], row['relations'],
                                  satisfiable_program, model_type='twoone'):
                count += 1
    logger.info('Number of unsatisfiable sentences: {}'.format(count))
    if count == 0:
        return 'satisfiable'
    if iteration >= max_iterations:
        return 'max_iter'
    return 'no'


def aggregate_answer_sets(answer_sets, how):
    assert how in ['random', 'intersection']
    if not answer_sets:
        return []
    if how == 'random':
        return random.choice(answer_sets)
    else:
        inter = set(answer_sets[0])
        for answer_set in answer_sets:
            inter = inter.intersection(answer_set)
        return inter


def unite_atoms(outputs, aggregation):
    if aggregation == 'intersection':
        output = aggregate_answer_sets(outputs, 'intersection')
    elif aggregation == 'random':
        output = aggregate_answer_sets(outputs, 'random')
    elif aggregation == 'weighted':
        output = aggregate_answer_sets(outputs, 'random')
    else:
        raise ValueError('Wrong aggregation value')

    if len(output) == 0:
        return [], [], []
    # Compute weight
    eweights = []
    rweights = []

    for atom in output:
        weight = 0
        for answer_set in outputs:
            if atom + '.' in answer_set:
                weight += 1
        weight = weight / len(outputs)
        if match_form(atom) == 'entity':
            eweights.append(weight)
        else:
            rweights.append(weight)
    assert len(output) == len(eweights) + len(rweights)
    return output, eweights, rweights

