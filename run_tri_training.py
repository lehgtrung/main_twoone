from methods.tri_training.tri_training import tri_training
import logging
import os
import argparse
import shutil
import json
from logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run tri training')
    parser.add_argument('--dataset',
                        action='store',
                        required=True)
    parser.add_argument('--fold',
                        type=int,
                        action='store',
                        required=True)
    parser.add_argument('--percent',
                        required=True,
                        type=int,
                        action='store')
    parser.add_argument('--with_disagreement',
                        required=True,
                        type=bool,
                        action='store')
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold
    percent = args.percent
    if args.with_disagreement:
        method = 'tri_training'
    else:
        method = 'tri_training_with_disagreement'

    LABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/labeled.json'
    TEMP_LABELED_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/temp_labeled.json'
    UNLABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/unlabeled.json'
    PREDICTION_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/prediction.json'
    AGREEMENT_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/agreement.json'
    LABELED_MODEL_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/models/labeled'
    WITH_DISAGREEMENT = args.with_disagreement
    LOG_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/logs.txt'

    configs = {
        'LABELED_PATH': LABELED_PATH,
        'UNLABELED_PATH': UNLABELED_PATH,
        'PREDICTION_PATH': PREDICTION_PATH,
        'TEMP_LABELED_PATH': TEMP_LABELED_PATH,
        'AGREEMENT_PATH': AGREEMENT_PATH,
        'LABELED_MODEL_PATH': LABELED_MODEL_PATH,
        'WITH_DISAGREEMENT': WITH_DISAGREEMENT,
        'LOG_PATH': LOG_PATH
    }

    with open('configs.json', 'w') as f:
        json.dump(configs, f)

    with open(configs['LOG_PATH'], 'w') as f:
        f.write('Start training\n')
    logger = Logger(path=configs['LOG_PATH'])

    tri_training(labeled_path=configs['LABELED_PATH'],
                 unlabeled_path=configs['UNLABELED_PATH'],
                 prediction_path=configs['PREDICTION_PATH'],
                 agreement_path=configs['AGREEMENT_PATH'],
                 temp_labeled_path=configs['TEMP_LABELED_PATH'],
                 labeled_model_path=configs['LABELED_MODEL_PATH'],
                 logger=logger,
                 log_path=configs['LOG_PATH'],
                 with_disagreement=configs['WITH_DISAGREEMENT'])


