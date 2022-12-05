from methods.self_training.self_training import self_training
import logging
import os
import argparse
import shutil
import json
from logger import Logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run self training')
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
    parser.add_argument('--start_iter',
                        required=False,
                        default=0,
                        type=int,
                        action='store')
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold
    percent = args.percent
    method = 'self_training'

    LABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/labeled.json'
    TEMP_LABELED_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/temp_labeled.json'
    UNLABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/unlabeled.json'
    PREDICTION_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/prediction.json'
    SELECTED_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/selected.json'
    LABELED_MODEL_PATH = './datasets/methods/{method}/{dataset}_{percent}/fold={fold}/models/iter={iter}/labeled'
    LABELED_MODEL_PATH = LABELED_MODEL_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    LOG_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/logs.txt'

    configs = {
        'LABELED_PATH': LABELED_PATH,
        'UNLABELED_PATH': UNLABELED_PATH,
        'PREDICTION_PATH': PREDICTION_PATH,
        'SELECTED_PATH': SELECTED_PATH,
        'LABELED_MODEL_PATH': LABELED_MODEL_PATH,
        'LOG_PATH': LOG_PATH
    }

    # Create paths
    os.makedirs(f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/models', exist_ok=True)

    with open('configs.json', 'w') as f:
        json.dump(configs, f)

    logger = Logger(path=configs['LOG_PATH'])

    self_training(labeled_path=configs['LABELED_PATH'],
                  unlabeled_path=configs['UNLABELED_PATH'],
                  prediction_path=configs['PREDICTION_PATH'],
                  selected_path=configs['SELECTED_PATH'],
                  labeled_model_path=configs['LABELED_MODEL_PATH'],
                  logger=logger,
                  log_path=configs['LOG_PATH'],
                  start_iter=args.start_iter)


