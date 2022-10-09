from methods.pluc.pluc import pseudo_labelling_under_curriculum
import logging
import os
import argparse
import shutil
import json
from logger import Logger


if __name__ == '__main__':
    fold = 1
    percent = 30
    LABELED_PATH = f'./datasets/core_conll04/conll04_{percent}/fold={fold}/labeled.json'
    TEMP_LABELED_PATH = f'./datasets/methods/pluc/conll04_{percent}/fold={fold}/temp_labeled.json'
    UNLABELED_PATH = f'./datasets/core_conll04/conll04_{percent}/fold={fold}/unlabeled.json'
    PREDICTION_PATH = f'./datasets/methods/pluc/conll04_{percent}/fold={fold}/prediction.json'
    SELECTED_PATH = f'./datasets/methods/pluc/conll04_{percent}/fold={fold}/selected.json'
    LABELED_MODEL_PATH = f'./datasets/methods/pluc/conll04_{percent}/fold={fold}/models/labeled'
    INTERMEDIATE_MODEL_PATH = './datasets/methods/pluc/conll04_{percent}/fold={fold}/models/inter_{iteration}'.format(
        percent=percent,
        fold=fold,
        iteration='{iteration}'
    )
    LOG_PATH = f'./datasets/methods/pluc/conll04_{percent}/fold={fold}/logs.txt'

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

    pseudo_labelling_under_curriculum(labeled_path=configs['LABELED_PATH'],
                                      unlabeled_path=configs['UNLABELED_PATH'],
                                      prediction_path=configs['PREDICTION_PATH'],
                                      temp_labeled_path=configs['TEMP_LABELED_PATH'],
                                      selected_path=configs['SELECTED_PATH'],
                                      labeled_model_path=configs['LABELED_MODEL_PATH'],
                                      intermediate_model_path=configs['INTERMEDIATE_MODEL_PATH'],
                                      logger=logger,
                                      log_path=configs['LOG_PATH'])


