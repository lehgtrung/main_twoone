
from local_parser import local_parse
from models import *
from prediction_helper import make_prediction, load_model

# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)
import argparse


def train(args):
    Trainer = model.get_default_trainer_class()
    trainer = Trainer(
        model=model,
        train_path=f'{args.train_path}',
        test_path=DEFAULT_TEST_PATH,
        valid_path=DEFAULT_VALID_PATH,
        label_config=args.label_config,
        batch_size=int(args.batch_size),
        tag_form=args.tag_form,
        num_workers=0,
    )
    print("=== Start training ===")
    trainer.train_model(args=args)


def evaluate(args):
    Trainer = model.get_default_trainer_class()
    trainer = Trainer(
        model=model,
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        valid_path=DEFAULT_VALID_PATH,
        label_config=args.label_config,
        batch_size=int(args.batch_size),
        tag_form=args.tag_form,
        num_workers=0,
    )
    rets = trainer.evaluate_model(model=model, test_type='test')
    print(rets)


def predict(args):
    model = load_model(path=args.model_path)
    make_prediction(model=model, input_path=args.predict_input_path,
                    output_path=args.predict_output_path)


parser = argparse.ArgumentParser(description='Arguments for training.')
parser = local_parse(parser)
args = parser.parse_args()

# Constants
DEFAULT_TRAIN_PATH = f'./datasets/unified/train.{args.dataset}.json'
DEFAULT_TEST_PATH = f'./datasets/unified/test.{args.dataset}.json'
DEFAULT_VALID_PATH = f'./datasets/unified/valid.{args.dataset}.json'


if args.device is not None and args.device != 'cpu':
    torch.cuda.set_device(args.device)
elif args.device is None:
    if torch.cuda.is_available():
        gpu_idx, gpu_mem = set_max_available_gpu()
        args.device = f"cuda:{gpu_idx}"
    else:
        args.device = "cpu"


config = Config(**args.__dict__)
ModelClass = eval(args.model_class)
model = ModelClass(config)


if args.model_read_ckpt:
    print(f"reading params from {args.model_read_ckpt}")
    model = model.load(args.model_read_ckpt)
    model.token_embedding.token_indexing.update_vocab = False
elif args.token_emb_dim > 0 and args.pretrained_wv:
    print(f"reading pretrained wv from {args.pretrained_wv}")
    model.token_embedding.load_pretrained(args.pretrained_wv, freeze=True)
    model.token_embedding.token_indexing.update_vocab = False


if __name__ == '__main__':
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        raise RuntimeError('Invalid mode (only accept train/eval/test')

