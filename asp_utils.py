
import json
import subprocess
from tqdm import tqdm
import os
import glob
import random
import re
import torch
import copy
from collections import Counter
import shutil

# atomed_output_path = 'methods/tri_training_with_asp/atomed_preds/{iter_number}/{model_number}/{sent_number}.txt'
# answerset_output_path = 'methods/tri_training_with_asp/answersets/{iter_number}/{model_number}/{sent_number}.txt'
# command = 'clingo --opt-mode=optN methods/tri_training_with_asp/p5_norelation.lp ' + atomed_output_path + \
#           ' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'
GLOBAL_ATOMED_OUTPUT_PATH = './datasets/methods/{method}/{dataset}_{percent}/' \
                            'fold={fold}/iter={iter}/atomed_preds/{model_number}/{sent_number}.txt'
GLOBAL_ANSWERSET_OUTPUT_PATH = './datasets/methods/{method}/{dataset}_{percent}/' \
                            'fold={fold}/iter={iter}/answersets/{model_number}/{sent_number}.txt'



def solve(command, sent_number, maximal=True):
    # Write the program to a file
    process = subprocess.Popen(command.format(sent_number=sent_number).split(),
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    result = [e.split() for e in output.decode().split('\n')[:-2]]
    if maximal:
        max_len = max(len(e) for e in result)
        result = [e for e in result if len(e) == max_len]
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


def convert_to_consistent_answersets(preds_path, iter_number, model_number, configs):
    atomed_output_path = GLOBAL_ATOMED_OUTPUT_PATH.format(
        method=configs['method'],
        dataset=configs['dataset'],
        percent=configs['percent'],
        fold=configs['fold'],
        iter=iter_number,
        model_number=model_number,
        sent_number='{sent_number}'
    )
    answerset_output_path = GLOBAL_ANSWERSET_OUTPUT_PATH.format(
        method=configs['method'],
        dataset=configs['dataset'],
        percent=configs['percent'],
        fold=configs['fold'],
        iter=iter_number,
        model_number=model_number,
        sent_number='{sent_number}'
    )
    # command = 'clingo --opt-mode=optN asp_v2/v5/p5_with_rules.lp ' + atomed_output_path + \
    #           ' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'
    command = 'clingo --opt-mode=optN asp_v2/v5/p5_maximal.lp ' + atomed_output_path + \
              ' --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2'
    if os.path.exists(os.path.dirname(atomed_output_path.format(sent_number=0))):
        shutil.rmtree(os.path.dirname(atomed_output_path.format(sent_number=0)))
    if os.path.exists(os.path.dirname(answerset_output_path.format(sent_number=0))):
        shutil.rmtree(os.path.dirname(answerset_output_path.format(sent_number=0)))
    os.makedirs(os.path.dirname(atomed_output_path.format(sent_number=0)), exist_ok=True)
    os.makedirs(os.path.dirname(answerset_output_path.format(sent_number=0)), exist_ok=True)
    # Load the predictions
    with open(preds_path, 'r') as f:
        preds = json.load(f)
    # Convert them into atomed form
    for i, row in enumerate(preds):
        atoms = convert_to_atoms(row, prefix='atom')
        # Path to write the atomed preds down
        path = atomed_output_path.format(sent_number=i)
        write_down_a_list(path, atoms)
    # Convert the atomed preds to answersets
    for i in tqdm(range(len(preds))):
        path = answerset_output_path.format(sent_number=i)
        answersets = solve(command=command,
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
            'entities': [list(e) for e in entities],
            'relations': [list(r) for r in relations]
        })
    return data


def get_metrics(sent_list, preds_list, labels_list):
    n_correct, n_pred, n_label = 0, 0, 0
    i_count = 0
    for sent, preds, labels in zip(sent_list, preds_list, labels_list):
        preds = set(preds)
        labels = {tuple(x) for x in labels}

        n_pred += len(preds)
        n_label += len(labels)
        n_correct += len(preds & labels)

        i_count += 1

    precision = n_correct / (n_pred + 1e-8)
    recall = n_correct / (n_label + 1e-8)
    f1 = 2 / (1 / (precision + 1e-8) + 1 / (recall + 1e-8) + 1e-8)

    return precision, recall, f1


def evaluate_model(preds, gts, logger=None):
    with torch.no_grad():
        sents = []
        pred_entities = []
        pred_relations = []
        pred_relations_wNER = []
        label_entities = []
        label_relations = []
        label_relations_wNER = []
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            pred['entities'] = [[tuple(e) for e in pred['entities']]]
            gt['entities'] = [gt['entities']]
            pred['relations'] = [set(tuple(e) for e in pred['relations'])]
            gt['relations'] = [gt['relations']]
            # pred_span_to_etype = [{(ib, ie): etype for ib, ie, etype in x} for x in pred['entities']]
            # label_span_to_etype = [{(ib, ie): etype for ib, ie, etype in x} for x in gt['entities']]
            pred_entities += pred['entities']
            label_entities += gt['entities']
            pred_relations += pred['relations']
            label_relations += gt['relations']

            # pred_relations_wNER += [
            #     [
            #         (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
            #     ] for x, m in zip(pred['relations'], pred_span_to_etype)
            # ]
            # label_relations_wNER += [
            #     [
            #         (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
            #     ] for x, m in zip(gt['relations'], label_span_to_etype)
            # ]

            sents += [pred['tokens']]

        rets = {}
        rets['entity_p'], rets['entity_r'], rets['entity_f1'] = get_metrics(
            sents, pred_entities, label_entities)
        rets['relation_p'], rets['relation_r'], rets['relation_f1'] = get_metrics(
            sents, pred_relations, label_relations)
        # rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER'] = get_metrics(
        #     sents, pred_relations_wNER, label_relations_wNER)

    e_precision, e_recall, e_f1 = rets['entity_p'], rets['entity_r'], rets['entity_f1']
    print(f">> entity prec:{e_precision:.4f}, rec:{e_recall:.4f}, f1:{e_f1:.4f}")
    r_precision, r_recall, r_f1 = rets['relation_p'], rets['relation_r'], rets['relation_f1']
    print(f">> relation prec:{r_precision:.4f}, rec:{r_recall:.4f}, f1:{r_f1:.4f}")
    # rwe_precision, rwe_recall, rwe_f1 = rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER']
    # print(f">> relation with NER prec:{rwe_precision:.4f}, rec:{rwe_recall:.4f}, f1:{rwe_f1:.4f}")

    if logger:
        logger.info(
            f">> entity prec:{e_precision:.4f}, rec:{e_recall:.4f}, f1:{e_f1:.4f}\n"
            f">> relation prec:{r_precision:.4f}, rec:{r_recall:.4f}, f1:{r_f1:.4f}\n"
            # f">> relation prec:{rwe_precision:.4f}, rec:{rwe_recall:.4f}, f1:{rwe_f1:.4f}"
        )


def select_agreement_with_asp(iter_number, model_number1, model_number2,
                              unlabeled_path, out_path, configs):
    answerset_output_path = GLOBAL_ANSWERSET_OUTPUT_PATH.format(
        method=configs['method'],
        dataset=configs['dataset'],
        percent=configs['percent'],
        fold=configs['fold'],
        iter=iter_number,
        model_number='{model_number}',
        sent_number='{sent_number}'
    )
    with open(unlabeled_path, 'r') as f:
        unlabeled_data = json.load(f)
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
            selected_indices.append(i)
            choice = random.choice(range(len(intersection)))
            selected_answersets.append(intersection[choice])
            tokens_list.append(unlabeled_data[i]['tokens'])

    # Convert selected sentences
    gts = [unlabeled_data[i] for i in selected_indices]
    preds = convert_atoms_to_file_form(tokens_list, selected_answersets)
    with open(out_path, 'w') as f:
        json.dump(preds, f)
    agree_ratio = len(selected_indices) / len(unlabeled_data)
    return selected_indices, agree_ratio


def convert_one_answersets_to_file_form(unlabeled_path,
                                        iter_number,
                                        model_number,
                                        out_path,
                                        configs):
    answerset_output_path = GLOBAL_ANSWERSET_OUTPUT_PATH.format(
        method=configs['method'],
        dataset=configs['dataset'],
        percent=configs['percent'],
        fold=configs['fold'],
        iter=iter_number,
        model_number='{model_number}',
        sent_number='{sent_number}'
    )
    with open(unlabeled_path, 'r') as f:
        unlabeled_data = json.load(f)
    meta_paths = glob.glob(answerset_output_path.format(iter_number=iter_number,
                                                        model_number=model_number,
                                                        sent_number='*'))
    tokens_list = []
    selected_answersets = []
    indices = []
    for path in meta_paths:
        i = int(os.path.basename(path).split('.')[0])
        all_answersets = parse_answersets_from_file(path, with_break=True)
        # choice = random.choice(range(len(all_answersets)))
        tokens_list.append(unlabeled_data[i]['tokens'])
        selected_answersets.append(all_answersets[0])
        indices.append(i)
    preds = convert_atoms_to_file_form(tokens_list, selected_answersets)
    with open(out_path, 'w') as f:
        json.dump(preds, f)
    return indices


def calc_symbol_freq(symbols, n, threshold=0.5):
    counter = Counter(map(tuple, symbols))
    final_symbols = []
    for symbol in symbols:
        if counter[tuple(symbol)]/n >= threshold:
            final_symbols.append(symbol)
    return final_symbols


def collect_symbols(preds, i, field):
    collection = []
    for pred in preds:
        collection.extend(pred[i][field])
    return collection


def aggregate_on_symbols(model_paths):
    preds = []
    outputs = []
    n = len(model_paths)
    for path in model_paths:
        with open(path, 'r') as f:
            preds.append(json.load(f))
    for i in range(len(preds[0])):
        tokens = preds[0][i]['tokens']
        symbols = collect_symbols(preds, i, 'entities')
        entities = calc_symbol_freq(symbols, n)
        symbols = collect_symbols(preds, i, 'relations')
        relations = calc_symbol_freq(symbols, n)
        outputs.append({
            'tokens': tokens,
            'entities': [tuple(e) for e in entities],
            'relations': [tuple(r) for r in relations]
        })
    return outputs

