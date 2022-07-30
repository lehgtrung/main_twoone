from asp.asp_main import curriculum_training


if __name__ == '__main__':
    LABELED_PATH = './datasets/unified/train.CoNLL04_30_labeled.json'
    UNLABELED_PATH = './datasets/unified/train.CoNLL04_30_unlabeled.json'
    RAW_PSEUDO_LABELED_PATH = './datasets/pseudo/raw.CoNLL04_30.json'
    SELECTED_PSEUDO_LABELED_PATH = './datasets/pseudo/selected.CoNLL04_30.json'
    UNIFIED_PSEUDO_LABELED_PATH = './datasets/pseudo/unified.CoNLL04_30.json'
    LABELED_MODEL_PATH = './ckpts/pseudo/labeled/labeled'
    INTERMEDIATE_MODEL_PATH = './ckpts/pseudo/intermediate/intermediate'

    curriculum_training(labeled_path=LABELED_PATH,
                        unlabeled_path=UNLABELED_PATH,
                        raw_pseudo_labeled_path=RAW_PSEUDO_LABELED_PATH,
                        selected_pseudo_labeled_path=SELECTED_PSEUDO_LABELED_PATH,
                        unified_pseudo_labeled_path=UNIFIED_PSEUDO_LABELED_PATH,
                        labeled_model_path=LABELED_MODEL_PATH,
                        intermediate_model_path=INTERMEDIATE_MODEL_PATH)