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
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold
    percent = args.percent

    LABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/labeled.json'
    TEMP_LABELED_PATH = f'./datasets/methods/self_training/{dataset}_{percent}/fold={fold}/temp_labeled.json'
    UNLABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/unlabeled.json'
    PREDICTION_PATH = f'./datasets/methods/self_training/{dataset}_{percent}/fold={fold}/prediction.json'
    SELECTED_PATH = f'./datasets/methods/self_training/{dataset}_{percent}/fold={fold}/selected.json'
    LABELED_MODEL_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/models/labeled'
    INTERMEDIATE_MODEL_PATH = './datasets/methods/self_training/{dataset}_{percent}/fold={fold}/models/inter_{iteration}'.format(
        dataset=dataset,
        percent=percent,
        fold=fold,
        iteration='{iteration}'
    )
    LOG_PATH = f'./datasets/methods/self_training/{dataset}_{percent}/fold={fold}/logs.txt'

    configs = {
        'LABELED_PATH': LABELED_PATH,
        'UNLABELED_PATH': UNLABELED_PATH,
        'PREDICTION_PATH': PREDICTION_PATH,
        'TEMP_LABELED_PATH': TEMP_LABELED_PATH,
        'SELECTED_PATH': SELECTED_PATH,
        'LABELED_MODEL_PATH': LABELED_MODEL_PATH,
        'INTERMEDIATE_MODEL_PATH': INTERMEDIATE_MODEL_PATH,
        'LOG_PATH': LOG_PATH
    }

    with open('configs.json', 'w') as f:
        json.dump(configs, f)

    with open(configs['LOG_PATH'], 'w') as f:
        f.write('Start training\n')
    logger = Logger(path=configs['LOG_PATH'])

    self_training(labeled_path=configs['LABELED_PATH'],
                  unlabeled_path=configs['UNLABELED_PATH'],
                  prediction_path=configs['PREDICTION_PATH'],
                  temp_labeled_path=configs['TEMP_LABELED_PATH'],
                  selected_path=configs['SELECTED_PATH'],
                  labeled_model_path=configs['LABELED_MODEL_PATH'],
                  intermediate_model_path=configs['INTERMEDIATE_MODEL_PATH'],
                  logger=logger,
                  log_path=configs['LOG_PATH'])


