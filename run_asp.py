from asp.asp_main import curriculum_training
import logging
import os
import argparse
import shutil


def check_data(dataset):
    for fold in range(1, 11):
        _path1 = './datasets/{dataset}/folds/{fold}/labeled.json'.format(dataset=dataset, fold=fold)
        _path2 = './datasets/{dataset}/folds/{fold}/unlabeled.json'.format(dataset=dataset, fold=fold)
        if not os.path.exists(_path1) or not os.path.exists(_path2):
            raise ValueError('Dataset not exist: ', _path1)


def create_folder_for_ssl(dataset, max_iter):
    for fold in range(1, 11):
        for agg in ['random', 'weighted', 'intersection']:
            for i in range(max_iter):
                _paths = [
                    './datasets/{dataset}/{fold}/train'.format(dataset=dataset, fold=fold),
                    './datasets/{dataset}/{fold}/{aggregation}/{iteration}'.format(dataset=dataset,
                                                                                   fold=fold,
                                                                                   aggregation=agg,
                                                                                   iteration=i),
                    './ckpts/{dataset}/{fold}/labeled'.format(dataset=dataset, fold=fold),
                    './ckpts/{dataset}/{fold}/raw'.format(dataset=dataset, fold=fold),
                    './ckpts/{dataset}/{fold}/{aggregation}/{iteration}/intermediate'.format(dataset=dataset,
                                                                                             fold=fold,
                                                                                             aggregation=agg,
                                                                                             iteration=i),
                    './logs/{dataset}/{fold}/{aggregation}/'.format(dataset=dataset,
                                                                    fold=fold,
                                                                    aggregation=agg)
                ]
                # Remove and re-create folders
                for _path in _paths:
                    # shutil.rmtree(_path)
                    os.makedirs(_path, exist_ok=True)


def set_conll04_arguments_asp(parser):
    print('READ ARGUMENTS FROM ASP')
    parser.add_argument('--aggregation',
                        action='store',
                        required=True)

    parser.add_argument('--fold',
                        required=True,
                        type=int,
                        action='store')

    parser.add_argument('--dataset',
                        required=True,
                        type=str,
                        action='store')

    return parser


parser = argparse.ArgumentParser(description='CONLL04')
parser = set_conll04_arguments_asp(parser)
args = parser.parse_args()

LABELED_PATH = './datasets/{dataset}/folds/{fold}/labeled.json'.format(dataset=args.dataset, fold=args.fold)
UNLABELED_PATH = './datasets/{dataset}/folds/{fold}/unlabeled.json'.format(dataset=args.dataset, fold=args.fold)
RAW_PSEUDO_LABELED_PATH = './datasets/{dataset}/{fold}/{aggregation}/{iteration}/raw.json'.format(
    dataset=args.dataset,
    fold=args.fold,
    aggregation=args.aggregation,
    iteration='{iteration}'
)
SELECTED_PSEUDO_LABELED_PATH = './datasets/{dataset}/{fold}/{aggregation}/{iteration}/selected.json'.format(
    dataset=args.dataset,
    fold=args.fold,
    aggregation=args.aggregation,
    iteration='{iteration}'
)
UNIFIED_PSEUDO_LABELED_PATH = './datasets/{dataset}/{fold}/{aggregation}/{iteration}/unified.json'.format(
    dataset=args.dataset,
    fold=args.fold,
    aggregation=args.aggregation,
    iteration='{iteration}'
)
LABELED_MODEL_PATH = './ckpts/{dataset}/{fold}/labeled/labeled'.format(
    dataset=args.dataset,
    fold=args.fold
)
RAW_MODEL_PATH = './ckpts/{dataset}/{fold}/raw/raw'.format(
    dataset=args.dataset,
    fold=args.fold
)
INTERMEDIATE_MODEL_PATH = './ckpts/{dataset}/{fold}/{aggregation}/{iteration}/intermediate/intermediate'.format(
    dataset=args.dataset,
    fold=args.fold,
    aggregation=args.aggregation,
    iteration='{iteration}'
)
LOG_PATH = './logs/{dataset}/{fold}/{aggregation}/log.txt'.format(
    dataset=args.dataset,
    fold=args.fold,
    aggregation=args.aggregation
)
AGGREGATION = args.aggregation

configs = {
    'LABELED_PATH': LABELED_PATH,
    'UNLABELED_PATH': UNLABELED_PATH,
    'RAW_PSEUDO_LABELED_PATH': RAW_PSEUDO_LABELED_PATH,
    'SELECTED_PSEUDO_LABELED_PATH': SELECTED_PSEUDO_LABELED_PATH,
    'UNIFIED_PSEUDO_LABELED_PATH': UNIFIED_PSEUDO_LABELED_PATH,
    'LABELED_MODEL_PATH': LABELED_MODEL_PATH,
    'RAW_MODEL_PATH': RAW_MODEL_PATH,
    'INTERMEDIATE_MODEL_PATH': INTERMEDIATE_MODEL_PATH,
    'LOG_PATH': LOG_PATH,
    'AGGREGATION': AGGREGATION
}

logging.basicConfig(filename=configs['LOG_PATH'], filemode='w',
                        format='%(asctime)s \n%(message)s\n',
                        datefmt='%b %d %Y %H:%M:%S',
                        level=logging.DEBUG)
logger = logging.getLogger()
logger.info(f'python run_asp.py --dataset {args.dataset} --fold {args.fold} --aggregation {args.aggregation}')


if __name__ == '__main__':

    # for key in configs:
    #     path = os.path.dirname(configs[key])
    #     if os.path.exists(path):
    #         os.makedirs(path, exist_ok=True)

    # create_folder_for_ssl(args.dataset, 5)
    # exit()

    print('Checking data')
    check_data(args.dataset)
    print('Data is ok')

    # Different ways to compute aggregation function: random, intersection, weighted
    # Number of iterations = 1, 2, 4, 6, 8, 10
    # Fix number of iterations, change aggregation function and vise versa
    # Change percentage: 10%, 30%, 50%, 70%, 90%
    # 3 datasets, 2 models
    # Record training time

    curriculum_training(labeled_path=configs['LABELED_PATH'],
                        unlabeled_path=configs['UNLABELED_PATH'],
                        raw_pseudo_labeled_path=configs['RAW_PSEUDO_LABELED_PATH'],
                        selected_pseudo_labeled_path=configs['SELECTED_PSEUDO_LABELED_PATH'],
                        unified_pseudo_labeled_path=configs['UNIFIED_PSEUDO_LABELED_PATH'],
                        labeled_model_path=configs['LABELED_MODEL_PATH'],
                        raw_model_path=configs['RAW_MODEL_PATH'],
                        intermediate_model_path=configs['INTERMEDIATE_MODEL_PATH'],
                        logger=logger,
                        aggregation=configs['AGGREGATION'],
                        max_iterations=5
                    )


