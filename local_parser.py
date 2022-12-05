def none_or_str(value):
    if value == 'None':
        return None
    return value


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


def set_conll04_arguments_main(parser):
    print('READ ARGUMENTS FROM MAIN')
    parser.add_argument('--mode',
                        default='train',
                        action='store', )

    parser.add_argument('--model_class',
                        default='None',
                        action='store', )

    parser.add_argument('--model_read_ckpt',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--model_write_ckpt',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--log_path',
                        default=None,
                        type=none_or_str,
                        action='store', )

    parser.add_argument('--pretrained_wv',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--dataset',
                        default='ACE05',
                        action='store', )

    parser.add_argument('--label_config',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--batch_size',
                        default=32, type=int,
                        action='store', )

    parser.add_argument('--evaluate_interval',
                        default=1000, type=int,
                        action='store', )

    parser.add_argument('--max_steps',
                        default=int(1e9), type=int,
                        action='store')

    parser.add_argument('--max_epoches',
                        default=100, type=int,
                        action='store')

    parser.add_argument('--decay_rate',
                        default=0.05, type=float,
                        action='store')

    #### Model Config
    parser.add_argument('--token_emb_dim',
                        default=100, type=int,
                        action='store', )

    parser.add_argument('--char_encoder',
                        default='lstm',
                        action='store', )

    parser.add_argument('--char_emb_dim',
                        default=0, type=int,
                        action='store', )

    parser.add_argument('--cased',
                        default=False, type=int,
                        action='store', )

    parser.add_argument('--hidden_dim',
                        default=200, type=int,
                        action='store', )

    parser.add_argument('--num_layers',
                        default=3, type=int,
                        action='store', )

    parser.add_argument('--crf',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--loss_reduction',
                        default='sum',
                        action='store', )

    parser.add_argument('--maxlen',
                        default=None, type=int,
                        action='store', )

    parser.add_argument('--dropout',
                        default=0.5, type=float,
                        action='store', )

    parser.add_argument('--optimizer',
                        default='sgd',
                        action='store', )

    parser.add_argument('--lr',
                        default=0.02, type=float,
                        action='store', )

    parser.add_argument('--vocab_size',
                        default=500000, type=int,
                        action='store', )

    parser.add_argument('--vocab_file',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--ner_tag_vocab_size',
                        default=64, type=int,
                        action='store', )

    parser.add_argument('--re_tag_vocab_size',
                        default=128, type=int,
                        action='store', )

    parser.add_argument('--lm_emb_dim',
                        default=0, type=int,
                        action='store', )

    parser.add_argument('--lm_emb_path',
                        default='', type=str,
                        action='store', )

    parser.add_argument('--head_emb_dim',
                        default=0, type=int,
                        action='store', )

    parser.add_argument('--tag_form',
                        default='iob2',
                        action='store', )

    parser.add_argument('--warm_steps',
                        default=1000, type=int,
                        action='store', )

    parser.add_argument('--grad_period',
                        default=1, type=int,
                        action='store', )

    parser.add_argument('--device',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--train_path',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--predict_input_path',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--predict_output_path',
                        default=None, type=none_or_str,
                        action='store', )

    parser.add_argument('--test_type',
                        default=None, type=none_or_str,
                        action='store', )
    return parser


def conll04_script():
    train_script = """
        python -u ../main.py \
        --mode train \
        --num_layers 3 \
        --batch_size 8  \
        --evaluate_interval 500 \
        --dataset CoNLL04 \
        --pretrained_wv ../wv/glove.6B.100d.conll04.txt \
        --max_epoches 2000 \
        --max_steps 30000 \
        --model_class JointModel \
        --crf None  \
        --optimizer adam \
        --lr 0.001  \
        --tag_form iob2 \
        --cased 0  \
        --token_emb_dim 100 \
        --char_emb_dim 30 \
        --char_encoder lstm  \
        --lm_emb_dim 4096 \
        --head_emb_dim 768 \
        --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
        --hidden_dim 200     --ner_tag_vocab_size 9 \
        --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
        --grad_period 1 --warm_steps 1000 \
        --model_write_ckpt {model_write_ckpt} \
        --train_path {train_path}
        """
    predict_script = """
            python -u ../main.py
            --mode predict \
            --model_class JointModel \
            --model_read_ckpt {model_read_ckpt} \
            --predict_input_path {predict_input_path} \
            --predict_output_path {predict_output_path}
            """
    eval_script = """
            python -u ../main.py \
            --mode eval \
            --num_layers 3 \
            --batch_size 8  \
            --evaluate_interval 500 \
            --dataset CoNLL04 \
            --pretrained_wv ./wv/glove.6B.100d.conll04.txt \
            --max_epoches 2000 \
            --max_steps 30000 \
            --model_class JointModel \
            --crf None  \
            --optimizer adam \
            --lr 0.001  \
            --tag_form iob2 \
            --cased 0  \
            --token_emb_dim 100 \
            --char_emb_dim 30 \
            --char_encoder lstm  \
            --lm_emb_dim 4096 \
            --head_emb_dim 768 \
            --lm_emb_path ../wv/albert.conll04_with_heads.pkl \
            --hidden_dim 200     --ner_tag_vocab_size 9 \
            --re_tag_vocab_size 11     --vocab_size 15000     --dropout 0.5  \
            --grad_period 1 --warm_steps 1000 \
            --model_read_ckpt {model_read_ckpt}
    """
    CONLL04_SCRIPT = {
        'train': train_script,
        'eval': eval_script,
        'predict': predict_script
    }
    return CONLL04_SCRIPT


