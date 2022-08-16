from .asp_ult import *
from tqdm import tqdm
from .asp_converter import *
from .asp_checker import *

import numpy as np
import torch
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def conll04_script():
    train_script = """
        python -u ./main.py \
        --mode train \
        --num_layers 3 \
        --batch_size 8  \
        --evaluate_interval 1000 \
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
        --lm_emb_path ./wv/albert.conll04_with_heads.pkl \
        --hidden_dim 200     --ner_tag_vocab_size 9 \
        --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
        --grad_period 1 --warm_steps 1000 \
        --model_write_ckpt {model_write_ckpt} \
        --train_path {train_path}
        """
    predict_script = """
            python -u ./main.py \
            --mode predict \
            --model_class JointModel \
            --model_read_ckpt {model_read_ckpt} \
            --predict_input_path {predict_input_path} \
            --predict_output_path {predict_output_path}
            """
    eval_script = """
            python -u ./main.py \
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
            --lm_emb_path ./wv/albert.conll04_with_heads.pkl \
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


def verify_and_infer(entities, relations, inference_program, model_type):
    assert model_type in ['twoone', 'spert']
    if model_type == 'spert':
        entities = spert_to_twoone(entities, relations, 'entity')
        relations = spert_to_twoone(entities, relations, 'relation')
    final_outputs = []
    # Remove connected components
    e_atoms = convert_original_to_atoms(entities, 'entity')
    r_atoms = convert_original_to_atoms(relations, 'relation')
    program = concat_facts(e_atoms, r_atoms)
    answer_sets = solve_v2(program)
    for answer_set in answer_sets:
        es, rs = convert_solutions_back(answer_set)
        # Inference starts here
        program = inference_program + '\n' + concat_facts(es, rs)
        solution = solve(program)
        if not solution:
            continue
        solution = ['ok(' + atom + ')' for atom in solution]
        es, rs = convert_solutions_back(solution)
        # Inference ends here
        final_outputs.append(es + rs)
    return final_outputs, e_atoms + r_atoms


def verify_and_infer_file(input_path, output_path, aggregation):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    with open('asp/inference.lp') as f:
        inference_program = f.read()
    with open('asp/satisfiable.lp') as f:
        satisfiable_program = f.read()
    data_points = []
    for i, row in tqdm(enumerate(input_data), total=len(input_data)):
        tokens = row['tokens']
        entities = row['entities']
        relations = row['relations']

        # Adhoc debug
        if i == 740:
            continue

        # First, check if the prediction satisfiable
        satisfiable = is_satisfiable(entities, relations, satisfiable_program, model_type='twoone')

        # If NOT satisfiable
        if not satisfiable:
            print(f'{i}: unsatisfiable')

        if not satisfiable:
            final_outputs, atoms = verify_and_infer(entities, relations, inference_program, model_type='twoone')

            atoms = remove_wrap(atoms, wrap_type='atom')
            word_atoms = convert_position_to_word_atoms(tokens, atoms)

            # Unite atoms
            united_atoms, eweights, rweights = unite_atoms(final_outputs, aggregation)

            if len(united_atoms) == 0:
                print('Empty selection: ', word_atoms)

            data_point = convert_solution_to_data(tokens, united_atoms)
            data_point = {
                'tokens': data_point['tokens'],
                'entities': data_point['entities'],
                'relations': data_point['relations'],
                'id': i,
                'satisfiable': 0,
                'atoms': word_atoms,
                'eweights': eweights,
                'rweights': rweights
            }
        else:
            e_atoms = convert_original_to_atoms(entities, 'entity')
            r_atoms = convert_original_to_atoms(relations, 'relation')
            atoms = e_atoms + r_atoms
            atoms = remove_wrap(atoms, wrap_type='atom')
            word_atoms = convert_position_to_word_atoms(tokens, atoms)
            data_point = {
                'tokens': tokens,
                'entities': entities,
                'relations': relations,
                'id': i,
                'satisfiable': 1,
                'atoms': word_atoms,
                'eweights': [1.0 for _ in range(len(entities))],
                'rweights': [1.0 for _ in range(len(relations))]
            }
        data_points.append(data_point)
    with open(output_path, 'w') as f:
        json.dump(data_points, f)


def curriculum_training(labeled_path,
                        unlabeled_path,
                        raw_pseudo_labeled_path,
                        selected_pseudo_labeled_path,
                        unified_pseudo_labeled_path,
                        labeled_model_path,
                        raw_model_path,
                        intermediate_model_path,
                        logger,
                        aggregation,
                        max_iterations,
                        ):
    SCRIPT = conll04_script()
    TRAIN_SCRIPT = SCRIPT['train']
    PREDICT_SCRIPT = SCRIPT['predict']
    EVAL_SCRIPT = SCRIPT['eval']

    logger.info(f'Labeled path: {labeled_path}')
    logger.info(f'Aggregation function: {aggregation}')

    # Step 1: Train on labeled data
    if not model_exists(labeled_model_path):
        script = TRAIN_SCRIPT.format(model_write_ckpt=labeled_model_path,
                                     train_path=labeled_path)
        logger.info('Train on labeled data')
        subprocess.run(script, shell=True, check=True)
    else:
        logger.info('Labeled model exists, skip training ...')

    iteration = 0
    while True:
        formatted_raw_pseudo_labeled_path = raw_pseudo_labeled_path.format(iteration=iteration)
        formatted_raw_pseudo_labeled_path_bk = raw_pseudo_labeled_path.format(iteration=iteration) + '.bk'
        formatted_selected_pseudo_labeled_path = selected_pseudo_labeled_path.format(iteration=iteration)
        formatted_unified_pseudo_labeled_path = unified_pseudo_labeled_path.format(iteration=iteration)
        formatted_intermediate_model_path = intermediate_model_path.format(iteration=iteration)

        # Step 1: Predict on unlabeled data
        if iteration == 0:
            _path = labeled_model_path
        else:
            _path = intermediate_model_path.format(iteration=iteration-1)
        script = PREDICT_SCRIPT.format(model_read_ckpt=_path,
                                       predict_input_path=unlabeled_path,
                                       predict_output_path=formatted_raw_pseudo_labeled_path)
        logger.info('Round #{}: Predict on unlabeled data'.format(iteration))
        subprocess.run(script, shell=True, check=True)

        # Step 2: check convergence
        converged = check_convergence(iteration=iteration,
                                      max_iterations=max_iterations,
                                      raw_pseudo_labeled_path=formatted_raw_pseudo_labeled_path,
                                      logger=logger)
        if converged == 'satisfiable':
            logger.info('Round #{}: Converged by satisfiable'.format(iteration))
            break
        elif converged == 'max_iter':
            logger.info('Round #{}: Converged by max iteration'.format(iteration))
            break

        # Step 3: Train a model on raw prediction
        if iteration == 0:
            if not model_exists(raw_model_path):
                logger.info('Round #{}: Retrain on raw pseudo labels'.format(iteration))
                script = TRAIN_SCRIPT.format(model_write_ckpt=raw_model_path,
                                             train_path=formatted_raw_pseudo_labeled_path)
                subprocess.run(script, shell=True, check=True)
                # Make prediction from raw model and check number of unsatisfiable
                logger.info('Round #{}: Make prediction from raw model and check convergence'.format(iteration))
                script = PREDICT_SCRIPT.format(model_read_ckpt=raw_model_path,
                                               predict_input_path=unlabeled_path,
                                               predict_output_path=formatted_raw_pseudo_labeled_path_bk)
                subprocess.run(script, shell=True, check=True)
                check_convergence(iteration=iteration,
                                  max_iterations=max_iterations,
                                  raw_pseudo_labeled_path=formatted_raw_pseudo_labeled_path_bk,
                                  logger=logger)

        # Step 4: For each sentence, verify and infer => list of answer sets (ASs)
        logger.info('Round #{}: Verify, Infer and Select on pseudo-labeled data'.format(iteration))
        verify_and_infer_file(
            input_path=formatted_raw_pseudo_labeled_path,
            output_path=formatted_selected_pseudo_labeled_path,
            aggregation=aggregation
        )

        # Step 5 Unify labeled and selected pseudo labels
        logger.info('Round #{}: Unify labels and pseudo labels'.format(iteration))
        unify_two_datasets(labeled_path=labeled_path,
                           pseudo_path=formatted_selected_pseudo_labeled_path,
                           output_path=formatted_unified_pseudo_labeled_path)

        # Step 6: Retrain on labeled and pseudo-labeled data
        logger.info('Round #{}: Retrain on selected pseudo labels'.format(iteration))
        script = TRAIN_SCRIPT.format(model_write_ckpt=formatted_intermediate_model_path,
                                     train_path=formatted_unified_pseudo_labeled_path)
        subprocess.run(script, shell=True, check=True)
        # Eval the trained model
        script = EVAL_SCRIPT.format(model_read_ckpt=formatted_intermediate_model_path)
        subprocess.run(script, shell=True, check=True)

        iteration += 1




