import random
from asp_ult import *
from tqdm import tqdm
import glob, os


def conll04_script():
    train_script = """
        python -u ../main.py \
        --mode train \
        --num_layers 3 \
        --batch_size 8  \
        --evaluate_interval 500 \
        --dataset CoNLL04 \
        --pretrained_wv ../wv/glove.6B.100d.conll04.txt \
        --max_epoches 2000 \
        --max_steps 20000 \
        --model_class JointModel \
        --crf None  \
        --optimizer adam \
        --lr 0.001  \
        --tag_form iob2 \
        --cased 0  \
        --token_emb_dim 100 \
        --char_emb_dim 30 \
        --char_encoder lstm  \
        --lm_emb_dim 4096 \
        --head_emb_dim 768 \
        --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
        --hidden_dim 200     --ner_tag_vocab_size 9 \
        --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
        --grad_period 1 --warm_steps 1000 \
        --model_write_ckpt {model_write_ckpt} \
        --train_path {train_path}
        """
    predict_script = """
            python -u ../main.py \
            --mode predict \
            --model_class JointModel \
            --model_read_ckpt {model_read_ckpt} \
            --predict_input_path {predict_input_path} \
            --predict_output_path {predict_output_path}
            """
    eval_script = """
            python -u ../main.py \
            --mode eval \
            --num_layers 3 \
            --batch_size 8  \
            --evaluate_interval 500 \
            --dataset CoNLL04 \
            --pretrained_wv ./wv/glove.6B.100d.conll04.txt \
            --max_epoches 2000 \
            --max_steps 30000 \
            --model_class JointModel \
            --crf None  \
            --optimizer adam \
            --lr 0.001  \
            --tag_form iob2 \
            --cased 0  \
            --token_emb_dim 100 \
            --char_emb_dim 30 \
            --char_encoder lstm  \
            --lm_emb_dim 4096 \
            --head_emb_dim 768 \
            --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
            --hidden_dim 200     --ner_tag_vocab_size 9 \
            --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
            --grad_period 1 --warm_steps 1000 \
            --model_read_ckpt {model_read_ckpt}
    """
    CONLL04_SCRIPT = {
        'train': train_script,
        'eval': eval_script,
        'predict': predict_script
    }
    return CONLL04_SCRIPT


def convert_solution_to_data(tokens, solution):
    data_point = {
        'tokens': tokens,
        'entities': [],
        'relations': []
    }
    for atom in solution:
        if match_form(atom) == 'entity':
            entity_type, word = extract_from_atom(atom, 'entity')
            start, end = word.split('+')
            data_point['entities'].append([
                int(start),
                int(end),
                polish_type(entity_type)
            ])
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            data_point['relations'].append([
                int(hstart),
                int(hend),
                int(tstart),
                int(tend),
                polish_type(relation_type)
            ])
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


def verify_and_infer(entities, relations, inference_program):
    final_outputs = []
    # Remove connected components
    es = convert_original_to_atoms(entities, 'entity')
    rs = convert_original_to_atoms(relations, 'relation')
    program = concat_facts(es, rs)
    answer_sets = solve_v2(program)
    for answer_set in answer_sets:
        es, rs = convert_solutions_back(answer_set)
        program = inference_program + '\n' + concat_facts(es, rs)
        solution = solve(program)
        if not solution:
            continue
        solution = ['ok(' + atom + ')' for atom in solution]
        es, rs = convert_solutions_back(solution)
        final_outputs.append(es + rs)
    return final_outputs


def verify_and_infer_file(input_path, output_path):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    with open('inference.lp') as f:
        inference_program = f.read()
    data_points = []
    for i, row in tqdm(enumerate(input_data), total=len(input_data)):
        tokens = row['tokens']
        entities = row['entity_preds']
        relations = row['relation_preds']

        e_atoms = convert_original_to_atoms(entities, 'entity')
        r_atoms = convert_original_to_atoms(relations, 'relation')
        atoms = e_atoms + r_atoms

        final_outputs = verify_and_infer(entities, relations, inference_program)
        united_atoms = answer_sets_randomly_selection(final_outputs)
        if not united_atoms:
            print('Empty selection: ', atoms)

        data_point = convert_solution_to_data(tokens, united_atoms)
        data_point = {
            'tokens': data_point['tokens'],
            'entities': data_point['entities'],
            'relations': data_point['relations'],
            'id': i,
            'atoms': atoms
        }
        data_points.append(data_point)
    with open(output_path, 'w') as f:
        json.dump(data_points, f)


def answer_sets_randomly_selection(answer_sets):
    # Number of times an atom appears in each answer_set / total number of answer sets
    if not answer_sets:
        return []
    return random.choice(answer_sets)


def check_coverage(iteration):
    if iteration == 3:
        return True
    return False


def labeled_model_exists(path):
    if 'done.txt' in glob.glob(os.path.dirname(path)):
        return True
    return False


def unify_two_datasets(first_path, second_path, output_path):
    with open(first_path, 'r') as f:
        first = json.load(f)
    with open(second_path, 'r') as f:
        second = json.load(f)
    with open(output_path, 'w') as f:
        json.dump(first + second, f)


def curriculum_training(labeled_path,
                        unlabeled_path,
                        raw_pseudo_labeled_path,
                        selected_pseudo_labeled_path,
                        unified_pseudo_labeled_path,
                        labeled_model_path,
                        intermediate_model_path
                        ):
    SCRIPT = conll04_script()
    TRAIN_SCRIPT = SCRIPT['train']
    PREDICT_SCRIPT = SCRIPT['predict']

    # Step 1: Train on labeled data
    if not labeled_model_exists(labeled_model_path):
        script = TRAIN_SCRIPT.format(model_write_ckpt=labeled_model_path,
                                     train_path=labeled_path)
        print('Train on labeled data')
        #subprocess.run(script, shell=True, check=True)
    else:
        print('Labeled model exists')

    iteration = 1
    while True:
        # Step 2: Predict on unlabeled data
        script = PREDICT_SCRIPT.format(model_read_ckpt=labeled_model_path,
                                       predict_input_path=unlabeled_path,
                                       predict_output_path=raw_pseudo_labeled_path)
        print('Round #{}: Predict on unlabeled data'.format(iteration))
        subprocess.run(script, shell=True, check=True)

        # Step 3: For each sentence, verify and infer => list of answer sets (ASs)
        print('Round #{}: Verify, Infer and Select on pseudo-labeled data'.format(iteration))
        verify_and_infer_file(input_path=raw_pseudo_labeled_path,
                              output_path=selected_pseudo_labeled_path)

        # Step 3.5 Unify labeled and selected pseudo labels
        unify_two_datasets(first_path=selected_pseudo_labeled_path,
                           second_path=labeled_path,
                           output_path=unified_pseudo_labeled_path)

        # Step 4: Retrain on labeled and pseudo-labeled data
        print('Round #{}: Retrain on selected pseudo labels'.format(iteration))
        script = TRAIN_SCRIPT.format(model_write_ckpt=intermediate_model_path,
                                     train_path=unified_pseudo_labeled_path)
        subprocess.run(script, shell=True, check=True)

        # Step 5: return to Step 2 while not converge
        if check_coverage(iteration):
            break


if __name__ == '__main__':
    LABELED_PATH = '../datasets/unified/train.CoNLL04_30_labeled.json'
    UNLABELED_PATH = '../datasets/unified/train.CoNLL04_30_unlabeled.json'
    RAW_PSEUDO_LABELED_PATH = '../datasets/pseudo/raw.CoNLL04_30.json'
    SELECTED_PSEUDO_LABELED_PATH = '../datasets/pseudo/selected.CoNLL04_30.json'
    UNIFIED_PSEUDO_LABELED_PATH = '../datasets/pseudo/unified.CoNLL04_30.json'
    LABELED_MODEL_PATH = '../ckpts/pseudo/labeled/labeled'
    INTERMEDIATE_MODEL_PATH = '../ckpts/pseudo/intermediate/intermediate'

    curriculum_training(labeled_path=LABELED_PATH,
                        unlabeled_path=UNLABELED_PATH,
                        raw_pseudo_labeled_path=RAW_PSEUDO_LABELED_PATH,
                        selected_pseudo_labeled_path=SELECTED_PSEUDO_LABELED_PATH,
                        unified_pseudo_labeled_path=UNIFIED_PSEUDO_LABELED_PATH,
                        labeled_model_path=LABELED_MODEL_PATH,
                        intermediate_model_path=INTERMEDIATE_MODEL_PATH)



