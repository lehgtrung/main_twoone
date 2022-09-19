
from .asp_ult import format_for_asp, match_form, extract_from_atom, polish_type, concat_facts, solve


def convert_original_to_atoms(data, dtype, wrap=True):
    result = []
    for d in data:
        if dtype == 'entity':
            if wrap:
                e = 'atom({}("{}")).'.format(format_for_asp(d[2], 'entity'),
                                             str(d[0]) + '+' + str(d[1]))
            else:
                e = '{}("{}").'.format(format_for_asp(d[2], 'entity'),
                                       str(d[0]) + '+' + str(d[1]))
            result.append(e)
        else:
            if wrap:
                r = 'atom({}("{}","{}")).'.format(format_for_asp(d[4], 'relation'),
                                             str(d[0]) + '+' + str(d[1]), str(d[2]) + '+' + str(d[3]))
            else:
                r = '{}("{}","{}").'.format(format_for_asp(d[4], 'relation'),
                                            str(d[0]) + '+' + str(d[1]), str(d[2]) + '+' + str(d[3]))
            result.append(r)
    return result


def convert_position_to_word_atoms(tokens, atoms):
    word_atoms = []
    for atom in atoms:
        if match_form(atom) == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = word.split('+')
            _word = '_'.join(tokens[int(start):int(end)])
            word_atoms.append(
                f'{entity_type}("{_word}")'
            )
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            _head_word = '_'.join(tokens[int(hstart):int(hend)])
            _tail_word = '_'.join(tokens[int(tstart):int(tend)])
            word_atoms.append(
                f'{relation_type}("{_head_word}", "{_tail_word}")'
            )
    return word_atoms


def remove_wrap(atoms, wrap_type):
    assert wrap_type in ['atom', 'ok']
    new_atoms = []
    for atom in atoms:
        if wrap_type == 'atom':
            new_atoms.append(
                atom.replace('atom(', '')[:-1]
            )
        else:
            new_atoms.append(
                atom.replace('ok(', '')[:-1]
            )
    return new_atoms


def convert_position_atom_to_training_form(atom, atype=None):
    if not atype:
        if match_form(atom) == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = word.split('+')
            row = [int(start), int(end), polish_type(entity_type)]
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            row = [int(hstart), int(hend), int(tstart), int(tend), polish_type(relation_type)]
    else:
        if atype == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = word.split('+')
            row = [int(start), int(end), polish_type(entity_type)]
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            row = [int(hstart), int(hend), int(tstart), int(tend), polish_type(relation_type)]
    return row


def convert_solution_to_data(tokens, solution):
    data_point = {
        'tokens': tokens,
        'entities': [],
        'relations': []
    }
    for atom in solution:
        if match_form(atom) == 'entity':
            row = convert_position_atom_to_training_form(atom, 'entity')
            data_point['entities'].append(row)
        else:
            row = convert_position_atom_to_training_form(atom, 'relation')
            data_point['relations'].append(row)
    return data_point


def convert_solutions_back(solution):
    es = []
    rs = []
    for atom in solution:
        atom = atom.replace('ok(', '', 1).replace(')', '', 1) + '.'
        if atom.startswith('loc(') or atom.startswith('peop(') or \
                atom.startswith('org(') or atom.startswith('other('):
            es.append(atom)
        else:
            rs.append(atom)
    return es, rs


def spert_to_twoone(entities, relations, dtype):
    if dtype == 'entity':
        new_entities = []
        for entity in entities:
            new_entities.append(
                [entity['start'], entity['end'], entity['type']]
            )
        return new_entities
    else:
        new_relations = []
        for relation in relations:
            new_relations.append(
                [entities[relation['head']][0],
                 entities[relation['head']][1],
                 entities[relation['tail']][0],
                 entities[relation['tail']][1],
                 relation['type']]
            )
        return new_relations


def twoone_to_spert(entities, relations, dtype):
    if dtype == 'entity':
        new_entities = []
        for entity in entities:
            new_entities.append(
                {
                    'start': entity[0],
                    'end': entity[1],
                    'type': entity[2]
                }
            )
        return new_entities
    else:
        new_relations = []
        for relation in relations:
            for i, h_entity in enumerate(entities):
                if relation[0] == h_entity[0] and relation[1] == h_entity[1]:
                    for j, t_entity in enumerate(entities):
                        if relation[2] == t_entity[0] and relation[3] == t_entity[1]:
                            new_relations.append(
                                {
                                    'head': i,
                                    'tail': j,
                                    'type': relation[4]
                                }
                            )
        return new_relations


def is_satisfiable(entities, relations, satisfiable_program, model_type, wrap=False):
    assert model_type in ['twoone', 'spert']
    if model_type == 'spert':
        entities = spert_to_twoone(entities, relations, 'entity')
        relations = spert_to_twoone(entities, relations, 'relation')

    e_atoms = convert_original_to_atoms(entities, 'entity', wrap)
    r_atoms = convert_original_to_atoms(relations, 'relation', wrap)
    program = satisfiable_program + '\n' + concat_facts(e_atoms, r_atoms)
    solution = solve(program)
    if solution:
        return True
    return False


def is_inferable(entities, relations, inference_program, model_type, wrap=False):
    assert model_type in ['twoone', 'spert']
    if model_type == 'spert':
        entities = spert_to_twoone(entities, relations, 'entity')
        relations = spert_to_twoone(entities, relations, 'relation')

    e_atoms = convert_original_to_atoms(entities, 'entity', wrap)
    r_atoms = convert_original_to_atoms(relations, 'relation', wrap)
    program = inference_program + '\n' + concat_facts(e_atoms, r_atoms)
    solution = solve(program)
    return len(solution) > len(entities) + len(relations)

