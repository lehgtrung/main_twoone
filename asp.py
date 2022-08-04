from asp.asp_main import curriculum_training
import logging
import os
from local_parser import none_or_int, none_or_str
import argparse


def create_folder_for_ssl(dataset):
    for part in range(1, 11):
        for agg in ['random', 'weighted', 'intersection']:
            for i in range(5):
                _paths = [
                    './datasets/{dataset}/{part}/train'.format(dataset=dataset, part=part),
                    './datasets/{dataset}/{part}/{aggregation}/{iteration}'.format(dataset=dataset,
                                                                                   part=part,
                                                                                   aggregation=agg,
                                                                                   iteration=i),
                    './datasets/{dataset}/{part}/{aggregation}/{iteration}'.format(dataset=dataset,
                                                                                   part=part,
                                                                                   aggregation=agg,
                                                                                   iteration=i),
                    './ckpts/{dataset}/{part}/{aggregation}/labeled'.format(dataset=dataset,
                                                                            part=part,
                                                                            aggregation=agg),
                    './ckpts/{dataset}/{part}/{aggregation}/raw'.format(dataset=dataset,
                                                                        part=part,
                                                                        aggregation=agg),
                    './ckpts/{dataset}/{part}/{aggregation}/{iteration}/intermediate'.format(dataset=dataset,
                                                                                             part=part,
                                                                                             aggregation=agg,
                                                                                             iteration=i),
                    './logs/{dataset}/{part}/{aggregation}/{iteration}'.format(dataset=dataset,
                                                                               part=part,
                                                                               aggregation=agg,
                                                                               iteration=i)
                ]
                for _path in _paths:
                    os.makedirs(_path, exist_ok=True)


def set_conll04_arguments(parser):
    parser.add_argument('--aggregation',
                        action='store',
                        required=True)

    parser.add_argument('--part',
                        required=True,
                        type=int,
                        action='store')

    parser.add_argument('--dataset',
                        required=True,
                        type=str,
                        action='store')
    return parser


parser = argparse.ArgumentParser(description='CONLL04')
parser = set_conll04_arguments(parser)
args = parser.parse_args()


if __name__ == '__main__':
    LABELED_PATH = './datasets/{dataset}/{part}/train/labeled.json'.format(dataset=args.dataset, part=args.part)
    UNLABELED_PATH = './datasets/{dataset}/{part}/train/unlabeled.json'.format(dataset=args.dataset, part=args.part),
    RAW_PSEUDO_LABELED_PATH = './datasets/{dataset}/{part}/{aggregation}/{iteration}/raw.json'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation,
        iteration='{iteration}'
    )
    SELECTED_PSEUDO_LABELED_PATH = './datasets/{dataset}/{part}/{aggregation}/{iteration}/selected.json'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation,
        iteration='{iteration}'
    )
    UNIFIED_PSEUDO_LABELED_PATH = './datasets/{dataset}/{part}/{aggregation}/{iteration}/unified.json'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation,
        iteration='{iteration}'
    )
    LABELED_MODEL_PATH = './ckpts/{dataset}/{part}/{aggregation}/labeled/labeled'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation
    )
    RAW_MODEL_PATH = './ckpts/{dataset}/{part}/{aggregation}/raw/raw'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation
    )
    INTERMEDIATE_MODEL_PATH = './ckpts/{dataset}/{part}/{aggregation}/{iteration}/intermediate/intermediate'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation,
        iteration='{iteration}'
    )
    LOG_PATH = './logs/{dataset}/{part}/{aggregation}/{iteration}/log.txt'.format(
        dataset=args.dataset,
        part=args.part,
        aggregation=args.aggregation,
        iteration='{iteration}'
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

    # for key in configs:
    #     path = os.path.dirname(configs[key])
    #     if os.path.exists(path):
    #         os.makedirs(path, exist_ok=True)

    create_folder_for_ssl(args.dataset)

    # Different ways to compute aggregation function: random, intersection, weighted
    # Number of iterations = 1, 2, 4, 6, 8, 10
    # Fix number of iterations, change aggregation function and vise versa
    # Change percentage: 10%, 30%, 50%, 70%, 90%
    # 3 datasets, 2 models
    # Record training time

    # logging.basicConfig(filename=configs['LOG_PATH'], filemode='w',
    #                     format='%(asctime)s \n%(message)s',
    #                     datefmt='%b %d %Y %H:%M:%S',
    #                     level=logging.DEBUG)
    # logger = logging.getLogger()
    #
    # curriculum_training(labeled_path=configs['LABELED_PATH'],
    #                     unlabeled_path=configs['UNLABELED_PATH'],
    #                     raw_pseudo_labeled_path=configs['RAW_PSEUDO_LABELED_PATH'],
    #                     selected_pseudo_labeled_path=configs['SELECTED_PSEUDO_LABELED_PATH'],
    #                     unified_pseudo_labeled_path=configs['UNIFIED_PSEUDO_LABELED_PATH'],
    #                     labeled_model_path=configs['LABELED_MODEL_PATH'],
    #                     raw_model_path=configs['RAW_MODEL_PATH'],
    #                     intermediate_model_path=configs['INTERMEDIATE_MODEL_PATH'],
    #                     logger=logger,
    #                     aggregation=configs['AGGREGATION'],
    #                     max_iterations=4
    #                 )


