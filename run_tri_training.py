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
                        required=False,
                        default=0,
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
    if args.with_disagreement == 1:
        method = 'tri_training_with_disagreement'
    else:
        method = 'tri_training'

    LABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/labeled.json'
    TEMP_LABELED_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/temp_labeled.json'
    UNLABELED_PATH = f'./datasets/core_{dataset}/{dataset}_{percent}/fold={fold}/unlabeled.json'
    PREDICTION_PATH = './datasets/methods/{method}/{dataset}_{percent}/fold={fold}/iter={iter}/prediction.json'
    PREDICTION_PATH = PREDICTION_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    AGREEMENT_PATH = './datasets/methods/{method}/{dataset}_{percent}/fold={fold}/iter={iter}/agreement.json'
    AGREEMENT_PATH = AGREEMENT_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    LABELED_MODEL_PATH = './datasets/methods/{method}/{dataset}_{percent}/fold={fold}/models/iter={iter}/labeled'
    LABELED_MODEL_PATH = LABELED_MODEL_PATH.format(
        method=method,
        dataset=dataset,
        percent=percent,
        fold=fold,
        iter='{}'
    )
    WITH_DISAGREEMENT = args.with_disagreement
    LOG_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/logs.txt'

    VALID_PREDICTION_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/valid_preds.json'
    TEST_PREDICTION_PATH = f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/test_preds.json'

    configs = {
        'LABELED_PATH': LABELED_PATH,
        'UNLABELED_PATH': UNLABELED_PATH,
        'PREDICTION_PATH': PREDICTION_PATH,
        'TEMP_LABELED_PATH': TEMP_LABELED_PATH,
        'AGREEMENT_PATH': AGREEMENT_PATH,
        'LABELED_MODEL_PATH': LABELED_MODEL_PATH,
        'WITH_DISAGREEMENT': WITH_DISAGREEMENT,
        'LOG_PATH': LOG_PATH,
        'VALID_PREDICTION_PATH': VALID_PREDICTION_PATH,
        'TEST_PREDICTION_PATH': TEST_PREDICTION_PATH
    }

    # Create paths
    os.makedirs(f'./datasets/methods/{method}/{dataset}_{percent}/fold={fold}/models', exist_ok=True)

    with open('configs.json', 'w') as f:
        json.dump(configs, f)

    logger = Logger(path=configs['LOG_PATH'])

    tri_training(labeled_path=configs['LABELED_PATH'],
                 unlabeled_path=configs['UNLABELED_PATH'],
                 prediction_path=configs['PREDICTION_PATH'],
                 agreement_path=configs['AGREEMENT_PATH'],
                 temp_labeled_path=configs['TEMP_LABELED_PATH'],
                 labeled_model_path=configs['LABELED_MODEL_PATH'],
                 logger=logger,
                 log_path=configs['LOG_PATH'],
                 valid_prediction_path=configs['VALID_PREDICTION_PATH'],
                 test_prediction_path=configs['TEST_PREDICTION_PATH'],
                 start_iter=args.start_iter,
                 with_disagreement=configs['WITH_DISAGREEMENT'])


