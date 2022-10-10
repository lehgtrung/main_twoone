import random
import subprocess
import numpy as np
import torch
import json
import os
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
            --log_path {log_path}
    """
    CONLL04_SCRIPT = {
        'train': train_script,
        'eval': eval_script,
        'predict': predict_script
    }
    return CONLL04_SCRIPT


def add_suffix_to_path(path, suffix, split_by):
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    if split_by == '':
        mod_base_name = f'{base_name}_{suffix}'
    else:
        parts = base_name.split(split_by)
        mod_base_name = f'{parts[0]}_{suffix}{split_by}{parts[1]}'
    return os.path.join(dir_name, mod_base_name)


def tri_training(labeled_path,
                 unlabeled_path,
                 prediction_path,
                 temp_labeled_path,
                 selected_path,
                 labeled_model_path,
                 intermediate_model_path,
                 logger,
                 log_path,
                 delta=0.2):
    SCRIPT = conll04_script()
    TRAIN_SCRIPT = SCRIPT['train']
    PREDICT_SCRIPT = SCRIPT['predict']
    EVAL_SCRIPT = SCRIPT['eval']

    logger.info(f'Labeled path: {labeled_path}')
    # Step 0: Bootstrap sample 3 models
    boostrap_labeled_paths = []
    boostrap_labeled_model_paths = []
    for i in range(1, 4):
        boostrap_labeled_paths.append(add_suffix_to_path(labeled_path, suffix=i, split_by='.'))
        boostrap_labeled_model_paths.append(add_suffix_to_path(labeled_model_path, suffix=i, split_by=''))

    for i in range(1, 4):
        with open(labeled_path, 'r') as f:
            data = json.load(f)
            sample = np.random.choice(data, len(data))
        with open(boostrap_labeled_paths[i], 'w') as f:
            json.dump(sample, f)

    # Step 1: Train on labeled data
    for i in range(1, 4):
        if not model_exists(boostrap_labeled_model_paths[i]):
            script = TRAIN_SCRIPT.format(model_write_ckpt=boostrap_labeled_model_paths[i],
                                         train_path=boostrap_labeled_paths[i],
                                         log_path=log_path)
            logger.info(f'Train on labeled data on model #{i}')
            subprocess.run(script, shell=True, check=True)
        else:
            logger.info(f'Labeled model #{i} exists, skip training ...')

    for i in range(1, 4):
        ...
