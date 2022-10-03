from asp.asp_main import pseudo_labelling_under_curriculum
import logging
import os
import argparse
import shutil
import json
from logger import Logger


if __name__ == '__main__':
    LABELED_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/labeled.json'
    TEMP_LABELED_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/temp_labeled.json'
    UNLABELED_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/unlabeled.json'
    SELECTED_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/selected.json'
    LABELED_MODEL_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/models/labeled'
    INTERMEDIATE_MODEL_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/models/inter_{iteration}'
    LOG_PATH = './datasets/pseudo_labelling_under_curriculum/conll04_pct=30_fold=10/logs.txt'

    configs = {
        'LABELED_PATH': LABELED_PATH,
        'UNLABELED_PATH': UNLABELED_PATH,
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
                                      temp_labeled_path=configs['TEMP_LABELED_PATH'],
                                      selected_path=configs['SELECTED_PATH'],
                                      labeled_model_path=configs['LABELED_MODEL_PATH'],
                                      intermediate_model_path=configs['INTERMEDIATE_MODEL_PATH'],
                                      logger=logger,
                                      log_path=configs['LOG_PATH'])


