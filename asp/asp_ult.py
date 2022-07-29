import json
import subprocess
import ast

clingo_path = '/home/thoang/anaconda3/bin/clingo'
clingo_options = ['--outf=2', '-n 0']
clingo_command = [clingo_path] + clingo_options

drive_command = ['/home/thoang/anaconda3/bin/clingo', 'exp_area/drive55.py',
                 'exp_area/p1.lp', 'exp_area/p3.lp', '--outf=3']


def solve(program):
    input = program.encode()
    process = subprocess.Popen(clingo_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate(input)
    result = json.loads(output.decode())
    if result['Result'] == 'SATISFIABLE' or result['Result'] == 'OPTIMUM FOUND':
        solutions = [value['Value'] for value in result['Call'][0]['Witnesses']]
        return union_all_solutions(solutions)
    else:
        return None


def solve_v2(program):
    # Write the program to a file
    with open('exp_area/p3.lp', 'w') as f:
        f.write(program)
    process = subprocess.Popen(drive_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = ast.literal_eval(output.decode().split('\n')[-2])
    return result


def union_all_solutions(solutions):
    union = set()
    for solution in solutions:
        for atom in solution:
            if not atom.startswith('nOfOKAtoms'):
                union.add(atom)
    return list(union)


def concat_facts(es, rs):
    output = []
    for e in es:
        output.append(e)
    for r in rs:
        output.append(r)
    return '\n'.join(output)


def hash_entity(tokens, entity, with_atom):
    etype = format_for_asp(entity[2], 'entity')
    eword = '_'.join(tokens[entity[0]:entity[1]])
    if with_atom:
        return 'atom({}("{}")).'.format(etype, eword)
    return '{}("{}").'.format(etype, eword)


def hash_relation(tokens, relation, with_atom):
    rtype = format_for_asp(relation[4], 'relation')
    headword = '_'.join(tokens[relation[0]:relation[1]])
    tailword = '_'.join(tokens[relation[2]:relation[3]])
    if with_atom:
        return 'atom({}("{}","{}")).'.format(rtype, headword, tailword)
    return '{}("{}","{}").'.format(rtype, headword, tailword)


def match_form(atom):
    open_pos = atom.index('(')
    if atom[:open_pos] in ['peop', 'loc', 'org', 'other']:
        return 'entity'
    return 'relation'


def polish_type(atom_type):
    if atom_type in ['peop', 'loc', 'org', 'other']:
        return atom_type.capitalize()
    if atom_type == 'liveIn':
        return 'Live_In'
    elif atom_type == 'locatedIn':
        return 'Located_In'
    elif atom_type == 'orgbasedIn':
        return 'OrgBased_In'
    elif atom_type == 'workFor':
        return 'Work_For'
    return 'Kill'


def extract_from_atom(atom, form_type):
    open_pos = atom.index('(')
    close_pos = atom.index(')')
    if form_type == 'entity':
        return atom[:open_pos], atom[open_pos + 1:close_pos].strip().strip('"')
    # count number of comma
    count = atom.count(',')
    if count == 1:
        comma_pos = atom.index(',')
    else:
        comma_pos = atom.index('",') + 1
    return atom[:open_pos], \
           atom[open_pos + 1:comma_pos].strip().strip('"'), \
           atom[comma_pos + 1:close_pos].strip().strip('"')


def format_for_asp(s, type):
    if type == 'entity':
        return s.lower()
    else:
        splits = s.split('_')
        if len(splits) > 1:
            return '{}{}'.format(splits[0].lower(), splits[1].capitalize())
        return splits[0].lower()


def convert_original_to_atoms(data, dtype):
    result = []
    for d in data:
        if dtype == 'entity':
            e = 'atom({}("{}")).'.format(format_for_asp(d[2], 'entity'),
                                         str(d[0]) + '+' + str(d[1]))
            result.append(e)
        else:
            r = 'atom({}("{}","{}")).'.format(format_for_asp(d[4], 'relation'),
                                         str(d[0]) + '+' + str(d[1]), str(d[2]) + '+' + str(d[3]))
            result.append(r)
    return result