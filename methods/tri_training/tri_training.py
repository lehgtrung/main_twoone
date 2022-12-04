import random
import subprocess
import numpy as np
import torch
import json
import os
from asp_utils import evaluate_model, aggregate_on_symbols
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import copy


def conll04_script():
    train_script = """
        python -u ./main.py \
        --mode train \
        --num_layers 3 \
        --batch_size 8  \
        --evaluate_interval 500 \
        --dataset CoNLL04 \
        --pretrained_wv ./wv/glove.6B.100d.conll04.txt \
        --max_epoches 150 \
        --max_steps 3000000 \
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
        --grad_period 1 --warm_steps 300 \
        --model_write_ckpt {model_write_ckpt} \
        --train_path {train_path} \
        --log_path {log_path}
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
            --model_read_ckpt {model_read_ckpt} \
            --log_path {log_path}
    """
    CONLL04_SCRIPT = {
        'train': train_script,
        'eval': eval_script,
        'predict': predict_script
    }
    return CONLL04_SCRIPT


def check_size(path):
    with open(path, 'r') as f:
        return len(json.load(f))


def add_suffix_to_path(path, suffix, split_by):
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    if split_by == '':
        mod_base_name = f'{base_name}_{suffix}'
    else:
        parts = base_name.split(split_by)
        mod_base_name = f'{parts[0]}_{suffix}{split_by}{parts[1]}'
    return os.path.join(dir_name, mod_base_name)


def model_exists(path):
    if os.path.exists(os.path.join(os.path.dirname(path), os.path.basename(path) + '.pt')):
        return True
    return False


def transfer_data(in_path1, in_path2, out_path):
    with open(in_path1, 'r') as f:
        data1 = json.load(f)
    with open(in_path2, 'r') as f:
        data2 = json.load(f)
    with open(out_path, 'w') as f:
        json.dump(data1 + data2, f)


def select_agreement(in_path1, in_path2, in_path3, unlabeled_path,
                     out_path, logger, with_disagreement=False):
    with open(in_path1, 'r') as f:
        dataset1 = json.load(f)
    with open(in_path2, 'r') as f:
        dataset2 = json.load(f)
    with open(in_path3, 'r') as f:
        dataset3 = json.load(f)
    with open(unlabeled_path, 'r') as f:
        unlabeled_data = json.load(f)

    agreements = []
    dataset_size = len(dataset1)
    agreement_indices = []
    for i in range(dataset_size):
        entities1 = set([(e[0], e[1]) for e in dataset1[i]['entities']])
        relations1 = set([(e[0], e[1], e[2], e[3]) for e in dataset1[i]['relations']])
        entities2 = set([(e[0], e[1]) for e in dataset2[i]['entities']])
        relations2 = set([(e[0], e[1], e[2], e[3]) for e in dataset2[i]['relations']])
        entities3 = set([(e[0], e[1]) for e in dataset3[i]['entities']])
        relations3 = set([(e[0], e[1], e[2], e[3]) for e in dataset3[i]['relations']])

        if with_disagreement:
            if (entities1 == entities2 and relations1 == relations2) and \
                    (entities1 != entities3 or relations1 != relations3):
                agreements.append(dataset1[i])
                agreement_indices.append(i)
        else:
            if entities1 == entities2 and relations1 == relations2:
                agreements.append(dataset1[i])
                agreement_indices.append(i)
    gts = [unlabeled_data[i] for i in agreement_indices]
    for_eval_preds = copy.deepcopy(agreements)
    evaluate_model(for_eval_preds, gts, logger)
    with open(out_path, 'w') as f:
        json.dump(agreements, f)
    return agreement_indices, len(agreement_indices) / len(unlabeled_data)


def global_agreement_ratio(paths):
    datasets = []
    for path in paths:
        with open(path, 'r') as f:
            datasets.append(json.load(f))
    dataset_size = check_size(paths[0])
    agreement = 0
    for i in range(dataset_size):
        entities_set = []
        relations_set = []
        for j in range(len(paths)):
            entities = set([(e[0], e[1]) for e in datasets[j][i]['entities']])
            relations = set([(e[0], e[1], e[2], e[3]) for e in datasets[j][i]['relations']])
            entities_set.append(entities)
            relations_set.append(relations)
        flag = True
        for k in range(1, len(entities_set)):
            if entities_set[k] != entities_set[k-1]:
                flag = False
                break
        for k in range(1, len(relations_set)):
            if relations_set[k] != relations_set[k-1]:
                flag = False
                break
        if flag:
            agreement += 1
    return agreement / dataset_size


def percentage_correct(path):
    with open(path, 'r') as f:
        data = json.load(f)
    match = 0
    for row in data:
        entities = set([(e[0], e[1]) for e in row['entities']])
        entity_gts = set([(e[0], e[1]) for e in row['entity_gts']])

        relations = set([(e[0], e[1], e[2], e[3]) for e in row['relations']])
        relation_gts = set([(e[0], e[1], e[2], e[3]) for e in row['relation_gts']])

        if entities == entity_gts and relations == relation_gts:
            match += 1
    return match / len(data)


def report_f1(path, selected_indices, unlabeled_path, logger):
    with open(path, 'r') as f:
        preds = json.load(f)
    with open(unlabeled_path, 'r') as f:
        unlabeled_data = json.load(f)
    gts = [unlabeled_data[i] for i in selected_indices]
    evaluate_model(preds, gts, logger)


def tri_training(labeled_path,
                 unlabeled_path,
                 prediction_path,
                 agreement_path,
                 temp_labeled_path,
                 labeled_model_path,
                 logger,
                 log_path,
                 valid_prediction_path,
                 test_prediction_path,
                 with_disagreement,
                 start_iter=0,
                 max_iteration=7):
    SCRIPT = conll04_script()
    TRAIN_SCRIPT = SCRIPT['train']
    PREDICT_SCRIPT = SCRIPT['predict']
    EVAL_SCRIPT = SCRIPT['eval']

    DEFAULT_TEST_PATH = './datasets/core_conll04/test.conll04.json'
    DEFAULT_VALID_PATH = './datasets/core_conll04/valid.conll04.json'

    logger.info(f'Labeled path: {labeled_path}')
    # Step 0: Bootstrap sample 3 models
    boostrap_labeled_paths = []
    boostrap_labeled_model_paths = []
    boostrap_prediction_paths = []
    agreement_paths = []
    boostrap_temp_labeled_paths = []
    stop_update = [False for _ in range(3)]
    valid_prediction_paths = []
    test_prediction_paths = []
    for i in range(3):
        boostrap_labeled_paths.append(add_suffix_to_path(labeled_path, suffix=i, split_by='.'))
        boostrap_labeled_model_paths.append(add_suffix_to_path(labeled_model_path, suffix=i, split_by=''))
        boostrap_prediction_paths.append(add_suffix_to_path(prediction_path, suffix=i, split_by='.'))
        boostrap_temp_labeled_paths.append(add_suffix_to_path(temp_labeled_path, suffix=i, split_by='.'))
        agreement_paths.append(add_suffix_to_path(agreement_path, suffix=i, split_by='.'))
        valid_prediction_paths.append(add_suffix_to_path(valid_prediction_path, suffix=i, split_by='.'))
        test_prediction_paths.append(add_suffix_to_path(test_prediction_path, suffix=i, split_by='.'))

    if start_iter == 0:
        for i in range(3):
            with open(labeled_path, 'r') as f:
                data = json.load(f)
                sample = np.random.choice(data, len(data)).tolist()
            with open(boostrap_labeled_paths[i], 'w') as f:
                # first iteration
                logger.info(f'Boostrap #{i} size: {len(sample)}')
                json.dump(sample, f)

    # Step 1: Train on labeled data
    formatted_boostrap_labeled_model_paths = []
    for i in range(3):
        formatted_boostrap_labeled_model_paths.append(boostrap_labeled_model_paths[i].format(-1))
    for i in range(3):
        if not model_exists(boostrap_labeled_model_paths[i].format(-1)):
            os.makedirs(os.path.dirname(formatted_boostrap_labeled_model_paths[0]), exist_ok=True)

            script = TRAIN_SCRIPT.format(model_write_ckpt=formatted_boostrap_labeled_model_paths[i],
                                         train_path=boostrap_labeled_paths[i],
                                         log_path=log_path)
            logger.info(f'Train on labeled data on model #{i}')
            subprocess.run(script, shell=True, check=True)
            # Eval the trained model
            script = EVAL_SCRIPT.format(model_read_ckpt=formatted_boostrap_labeled_model_paths[i],
                                        log_path=log_path)
            subprocess.run(script, shell=True, check=True)
        else:
            if start_iter == -1:
                logger.info(f'Labeled model #{i} exists, skip training ...')

    iteration = start_iter
    while True:
        formatted_boostrap_prediction_paths = []
        formatted_agreement_paths = []
        formatted_boostrap_labeled_model_paths = []
        for i in range(3):
            formatted_boostrap_prediction_paths.append(boostrap_prediction_paths[i].format(iteration))
            formatted_agreement_paths.append(agreement_paths[i].format(iteration))
            formatted_boostrap_labeled_model_paths.append(boostrap_labeled_model_paths[i].format(iteration))
        os.makedirs(os.path.dirname(formatted_boostrap_prediction_paths[0]), exist_ok=True)
        os.makedirs(os.path.dirname(formatted_boostrap_labeled_model_paths[0]), exist_ok=True)
        if iteration == max_iteration:
            break
        # Step 2: make prediction for each model
        for i in range(3):
            script = PREDICT_SCRIPT.format(model_read_ckpt=boostrap_labeled_model_paths[i].format(iteration-1),
                                           predict_input_path=unlabeled_path,
                                           predict_output_path=formatted_boostrap_prediction_paths[i])
            logger.info(f'Round #{iteration}: Predict on unlabeled data on model m{i}')
            subprocess.run(script, shell=True, check=True)

        # Step 3: stop when predictions from differs under a small ratio
        agreement_ratio = global_agreement_ratio(formatted_boostrap_prediction_paths)
        logger.info(f'Round #{iteration}: Global agreement between 3 models: {agreement_ratio}')
        if agreement_ratio >= 0.9:
            logger.info(f'Round #{iteration}: Reach global agreement between 3 models')
            break

        # Step 4: otherwise, find agreements between models
        for i in range(2):
            for j in range(i+1, 3):
                if not stop_update[sum(range(3))-(i+j)]:
                    selected_indices, agree_ratio = select_agreement(
                        in_path1=formatted_boostrap_prediction_paths[i],
                        in_path2=formatted_boostrap_prediction_paths[j],
                        in_path3=formatted_boostrap_prediction_paths[sum(range(3))-(i+j)],
                        out_path=formatted_agreement_paths[sum(range(3))-(i+j)],
                        unlabeled_path=unlabeled_path,
                        logger=logger,
                        with_disagreement=with_disagreement
                    )
                    if agree_ratio >= 0.9:
                        stop_update[sum(range(3))-(i+j)] = True
                        logger.info(f'Round #{iteration}: Agreement ratio between model_{i} and model_{j}: '
                                    f'{round(agree_ratio * 100, 3)}, stop update')
                        logger.info(f'Round #{iteration}: Percent match of selected set: '
                                    f'{percentage_correct(formatted_agreement_paths[sum(range(3))-(i+j)])}')
                    logger.info('########################################################')
                    logger.info(f'Round #{iteration}: Agreement ratio between model_{i} and model_{j}: '
                                f'{round(agree_ratio*100, 3)}')
                    logger.info(f'Round #{iteration}: Percent match of selected set: '
                                f'{percentage_correct(formatted_agreement_paths[sum(range(3))-(i+j)])}')
                    logger.info(f'Round #{iteration}: Selection size: {len(selected_indices)}')
                    logger.info(f'Round #{iteration}: F1 on selection')
                    report_f1(path=formatted_agreement_paths[sum(range(3)) - (i + j)],
                              selected_indices=selected_indices,
                              unlabeled_path=unlabeled_path,
                              logger=logger)
                    logger.info('########################################################')

        # Step 5: transfer
        for i in range(3):
            transfer_data(in_path1=labeled_path,
                          in_path2=formatted_agreement_paths[i],
                          out_path=boostrap_temp_labeled_paths[i])

        # Step 6: train on transfer data
        for i in range(3):
            script = TRAIN_SCRIPT.format(model_write_ckpt=formatted_boostrap_labeled_model_paths[i],
                                         train_path=boostrap_temp_labeled_paths[i],
                                         log_path=log_path)
            logger.info(f'Round #{iteration}: Train on labeled data on model #{i}')
            subprocess.run(script, shell=True, check=True)

        # Step 7: aggregate 3 models and check performance
        for i in range(3):
            script = PREDICT_SCRIPT.format(model_read_ckpt=formatted_boostrap_labeled_model_paths[i],
                                           predict_input_path=DEFAULT_VALID_PATH,
                                           predict_output_path=valid_prediction_paths[i])
            logger.info(f'Round #{iteration}: Predict on valid data on model m{i}')
            subprocess.run(script, shell=True, check=True)

        for i in range(3):
            script = PREDICT_SCRIPT.format(model_read_ckpt=formatted_boostrap_labeled_model_paths[i],
                                           predict_input_path=DEFAULT_TEST_PATH,
                                           predict_output_path=test_prediction_paths[i])
            logger.info(f'Round #{iteration}: Predict on test data on model m{i}')
            subprocess.run(script, shell=True, check=True)

        logger.info('Check F1 on valid data')
        model_agg = aggregate_on_symbols(model_paths=valid_prediction_paths)
        evaluate_model(model_agg, json.load(open(DEFAULT_VALID_PATH)), logger)
        logger.info('Check F1 on test data')
        model_agg = aggregate_on_symbols(model_paths=test_prediction_paths)
        evaluate_model(model_agg, json.load(open(DEFAULT_TEST_PATH)), logger)
        iteration += 1













