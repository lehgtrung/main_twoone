
import subprocess
import json
import ast

command = 'clingo --opt-mode=optN asp_v2/v5/p5_2.lp analysis/v5/{sent_number}.txt ' \
          '--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'


def solve(sent_number):
    # Write the program to a file
    process = subprocess.Popen(command.format(sent_number=sent_number).split(),
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = [e.split() for e in output.decode().split('\n')[:-2]]
    return result


def parse_console_output():
    ...


def convert_to_atom(some_atom):
    return ''


def run_asp(set_atoms):
    return []


def regulate_predictions(preds, gts):
    # preds in the form [atom(locatedIn(X,P), 1),..]
    # gts in the form [locatedIn(X,P),..]
    for i, atom_set in enumerate(zip(preds, gts)):
        answer_set = run_asp(atom_set)
        answer_set = [convert_to_atom(e) for e in answer_set]
        with open('', 'w') as f:
            json.dump()


if __name__ == '__main__':
    # print(solve(1))
    for k in solve(1):
        print(k)

