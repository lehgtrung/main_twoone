import random
from asp_ult import *


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
                start,
                end,
                polish_type(entity_type)
            ])
        else:
            relation_type, head_word, tail_word = extract_from_atom(atom, 'relation')
            hstart, hend = head_word.split('+')
            tstart, tend = tail_word.split('+')
            data_point['relations'].append([
                hstart,
                hend,
                tstart,
                tend,
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
    ...


def select_answer_set_file(input_path, output_path):
    ...


def answer_sets_randomly_selection(answer_sets):
    # Number of times an atom appears in each answer_set / total number of answer sets
    return random.choice(answer_sets)


def check_coverage():
    return False


def curriculum_training(labeled_path, labeled_model_path, unlabeled_path):
    based_train_script = """
        python -u ./train.py \
        --num_layers 2 \
        --batch_size 2  \
        --dataset CoNLL04 \
        --pretrained_wv ./wv/glove.6B.100d.conll04.txt \
        --max_epoches 3000 \
        --max_steps {} \
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
        --lm_emb_path ./wv/albert.conll04_with_heads.pkl \
        --hidden_dim 200     --ner_tag_vocab_size 9 \
        --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
        --grad_period 1 --warm_steps 1000 \
        --model_write_ckpt {} \
        --train_path {}
        """
    predict_script = """python ../predict_script.py {} {} {}"""

    # Step 1: Train on labeled data
    script = based_train_script.format(20000, labeled_model_path, labeled_path)
    print('Train on labeled data')
    subprocess.run(script, shell=True, check=True)

    iteration = 1
    while True:
        # Step 2: Predict on unlabeled data
        script = predict_script.format('', '', '')
        print('Round #{}: Predict on unlabeled data'.format(iteration))
        subprocess.run(script, shell=True, check=True)

        # Step 3: For each sentence, verify and infer => list of answer sets (ASs)
        verify_and_infer_file('', '')

        # Step 4: For each sentence, select 1 answer set from ASs (randomly or easiest first)
        select_answer_set_file('', '')

        # Step 5: Retrain on labeled and pseudo-labeled data
        script = based_train_script.format(20000, labeled_model_path, labeled_path)
        subprocess.run(script, shell=True, check=True)

        # Step 6: return to Step 2 while not converge
        if check_coverage():
            break


if __name__ == '__main__':
    with open('exp_area/p_star.lp') as f:
        verification_program = f.read()

    with open('inference.lp') as f:
        inference_program = f.read()

    with open('../datasets/ssl_outputs/argmax_predicted.CoNLL04_30_unlabeled.json') as f:
        pred_data = json.load(f)

    with open('../datasets/unified/train.CoNLL04_30_unlabeled.json') as f:
        gt_data = json.load(f)

    assert len(pred_data) == len(gt_data)
    print('Length: ', len(gt_data))
    count_s_equal_t = 0
    count_false_true = 0
    count_p_equal_t = 0
    pred_iou = []
    solution_iou = []
    data_points = []
    for i, (pred_row, gt_row) in enumerate(zip(pred_data, gt_data)):
        print('=============================')
        print(i)

        tokens = gt_row['tokens']
        entities = pred_row['entity_preds']
        relations = pred_row['relation_preds']

        print(convert_original_to_atoms(entities, 'entity'))
        print(convert_original_to_atoms(relations, 'relation'))

        final_outputs = verify_and_infer(entities, relations, inference_program)

        united_atoms = answer_sets_randomly_selection(final_outputs)

        print(final_outputs)
        print(united_atoms)

        data_point = convert_solution_to_data(tokens, united_atoms)

        # Convert solution to new data
        data_point = {
            'tokens': data_point['tokens'],
            'entities': data_point['entities'],
            'relations': data_point['relations'],
            'id': i
        }
        data_points.append(data_point)

    # with open('../datasets/ssl_train_data/argmax_w_all_answersets_with_intersection_complete.json', 'w') as f:
    #     json.dump(data_points, f)
