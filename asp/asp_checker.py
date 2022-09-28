
import json
import random
import os
from .asp_converter import is_satisfiable, convert_position_atom_to_training_form
from .asp_ult import match_form


def model_exists(path):
    if os.path.exists(os.path.join(os.path.dirname(path), os.path.basename(path) + '.pt')):
        return True
    return False


def unify_two_datasets(labeled_path, pseudo_path, output_path, with_weight=False):
    with open(labeled_path, 'r') as f:
        labeled = json.load(f)
        for line in labeled:
            line['eweights'] = [1.0 for _ in range(len(line['entities']))]
            line['rweights'] = [1.0 for _ in range(len(line['relations']))]
    with open(pseudo_path, 'r') as f:
        pseudo = json.load(f)
        if with_weight:
            for line in pseudo:
                line['eweights'] = [1.0 for _ in range(len(line['entities']))]
                line['rweights'] = [1.0 for _ in range(len(line['relations']))]
    with open(output_path, 'w') as f:
        json.dump(labeled + pseudo, f)


def transfer_and_subtract_two_datasets(labeled_path, unlabeled_path, indices):
    with open(labeled_path, 'r') as f:
        labeled = json.load(f)
    with open(unlabeled_path, 'r') as f:
        unlabeled = json.load(f)
    selected = []
    remains = []
    for i, row in enumerate(unlabeled):
        if i in indices:
            selected.append(row)
        else:
            remains.append(row)
    with open(labeled_path, 'w') as f:
        json.dump(labeled + selected, f)
    with open(unlabeled_path, 'w') as f:
        json.dump(remains, f)


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


def unite_atoms(answer_sets, aggregation):
    # if aggregation == 'intersection':
    #     output = aggregate_answer_sets(outputs, 'intersection')
    # elif aggregation == 'random':
    #     output = aggregate_answer_sets(outputs, 'random')
    # elif aggregation == 'weighted':
    #     output = aggregate_answer_sets(outputs, 'random')
    # else:
    #     raise ValueError('Wrong aggregation value')

    # if len(output) == 0:
    #     return [], [], []
    # Compute weight

    if aggregation == 'intersection':
        answer_set = aggregate_answer_sets(answer_sets, 'intersection')
        entities = []
        relations = []
        eweights = []
        rweights = []
        for atom in answer_set:
            if match_form(atom) == 'entity':
                entity = convert_position_atom_to_training_form(atom, 'entity')
                entities.append(entity)
                eweights.append(1.0)
            else:
                relation = convert_position_atom_to_training_form(atom, 'relation')
                relations.append(relation)
                rweights.append(1.0)
        entities = [entities]
        relations = [relations]
    else:
        entities = []
        relations = []
        eweights = []
        rweights = []

        for answer_set in answer_sets:
            _entities = []
            _relations = []
            _eweights = []
            _rweights = []
            for atom in answer_set:
                weight = 0
                for other_answer_set in answer_sets:
                    if atom + '.' in other_answer_set:
                        weight += 1
                weight = weight / len(answer_sets)
                if match_form(atom) == 'entity':
                    entity = convert_position_atom_to_training_form(atom, 'entity')
                    _entities.append(entity)
                    _eweights.append(weight)
                else:
                    relation = convert_position_atom_to_training_form(atom, 'relation')
                    _relations.append(relation)
                    _rweights.append(weight)
            entities.append(_entities)
            relations.append(_relations)
            eweights.append(_eweights)
            rweights.append(_rweights)
    return entities, relations, eweights, rweights

