import random
import subprocess
import numpy as np
import torch
import json
import os
from asp_utils import evaluate_model, convert_to_consistent_answersets, convert_one_answersets_to_file_form
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


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
            --log_path {log_path} \
            --test_type {test_type}
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


def model_exists(path):
    if os.path.exists(os.path.join(os.path.dirname(path), os.path.basename(path) + '.pt')):
        return True
    return False


def transfer_and_subtract_two_datasets(labeled_path,
                                       prediction_path,
                                       temp_labeled_path,
                                       selected_path,
                                       indices):
    with open(labeled_path, 'r') as f:
        labeled = json.load(f)
    with open(prediction_path, 'r') as f:
        prediction = json.load(f)
    selected = []
    remains = []
    for i, row in enumerate(prediction):
        if i in indices:
            selected.append(row)
        else:
            remains.append(row)
    with open(selected_path, 'w') as f:
        json.dump(selected, f)
    with open(temp_labeled_path, 'w') as f:
        json.dump(labeled + selected, f)


def transfer_data(in_path1, in_path2, out_path):
    with open(in_path1, 'r') as f:
        data1 = json.load(f)
    with open(in_path2, 'r') as f:
        data2 = json.load(f)
    with open(out_path, 'w') as f:
        json.dump(data1 + data2, f)


def report_f1(path, selected_indices, unlabeled_path, logger):
    with open(path, 'r') as f:
        preds = json.load(f)
    with open(unlabeled_path, 'r') as f:
        unlabeled_data = json.load(f)
    gts = [unlabeled_data[i] for i in selected_indices]
    evaluate_model(preds, gts, logger)


def self_training_with_asp(labeled_path,
                           unlabeled_path,
                           prediction_path,
                           selected_path,
                           labeled_model_path,
                           logger,
                           log_path,
                           start_iter,
                           configs,
                           max_iteration=1):
    SCRIPT = conll04_script()
    TRAIN_SCRIPT = SCRIPT['train']
    PREDICT_SCRIPT = SCRIPT['predict']
    EVAL_SCRIPT = SCRIPT['eval']

    logger.info(f'Labeled path: {labeled_path}')

    # Step 1: Train on labeled data
    if not model_exists(labeled_model_path.format(-1)):
        os.makedirs(os.path.dirname(labeled_model_path.format(-1)), exist_ok=True)
        script = TRAIN_SCRIPT.format(model_write_ckpt=labeled_model_path.format(-1),
                                     train_path=labeled_path,
                                     log_path=log_path)
        logger.info('Train on labeled data')
        subprocess.run(script, shell=True, check=True)
    else:
        logger.info('Labeled model exists, skip training ...')

    logger.info('Evaluate labeled model on valid set ...')
    script = EVAL_SCRIPT.format(model_read_ckpt=labeled_model_path.format(-1),
                                log_path=log_path, test_type='valid')
    subprocess.run(script, shell=True, check=True)
    logger.info('Evaluate labeled model on test set ...')
    script = EVAL_SCRIPT.format(model_read_ckpt=labeled_model_path.format(-1),
                                log_path=log_path, test_type='test')
    subprocess.run(script, shell=True, check=True)

    iteration = start_iter
    while True:
        if iteration >= max_iteration:
            break

        # Step 2: Predict on unlabeled data
        script = PREDICT_SCRIPT.format(model_read_ckpt=labeled_model_path.format(iteration-1),
                                       predict_input_path=unlabeled_path,
                                       predict_output_path=prediction_path)
        logger.info(f'Round #{iteration}: Predict on unlabeled data')
        subprocess.run(script, shell=True, check=True)

        # Step 3: Convert to consistent atoms
        logger.info(f'Round #{iteration}: Convert raw prediction to answersets')
        convert_to_consistent_answersets(prediction_path,
                                         iter_number=iteration,
                                         model_number=0,
                                         configs=configs)

        # Step 3: Convert answersets back to training format
        logger.info(f'Round #{iteration}: Convert answersets back to data')
        selected_indices = convert_one_answersets_to_file_form(
            unlabeled_path=unlabeled_path,
            iter_number=iteration,
            model_number=0,
            out_path=prediction_path,
            configs=configs)
        logger.info(f'Round #{iteration}: F1 on selection')
        report_f1(path=prediction_path,
                  selected_indices=selected_indices,
                  unlabeled_path=unlabeled_path,
                  logger=logger)

        # Step 4: Unify labeled and selected pseudo labels
        logger.info(f'Round #{iteration}: Unify labels and pseudo labels')
        transfer_data(in_path1=labeled_path,
                      in_path2=prediction_path,
                      out_path=selected_path)
        # logger.info(f'Round #{iteration}: Percent match of selected set: {percentage_correct(selected_path)}')

        os.makedirs(os.path.dirname(labeled_model_path.format(iteration)), exist_ok=True)
        # Step 5: Retrain on labeled and pseudo-labeled data
        logger.info(f'Round #{iteration}: Retrain on selected pseudo labels')
        script = TRAIN_SCRIPT.format(model_write_ckpt=labeled_model_path.format(iteration),
                                     train_path=selected_path,
                                     log_path=log_path)
        subprocess.run(script, shell=True, check=True)
        # Eval the trained model
        logger.info('Evaluate labeled model on valid set ...')
        script = EVAL_SCRIPT.format(model_read_ckpt=labeled_model_path.format(iteration),
                                    log_path=log_path, test_type='valid')
        subprocess.run(script, shell=True, check=True)
        logger.info('Evaluate labeled model on test set ...')
        script = EVAL_SCRIPT.format(model_read_ckpt=labeled_model_path.format(iteration),
                                    log_path=log_path, test_type='test')
        subprocess.run(script, shell=True, check=True)

        iteration += 1

