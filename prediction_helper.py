from models import *


def make_prediction(model, input_path, output_path):
    kept_fields = ['entity_preds', 'relation_preds', 'table_probs']
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    outputs = []
    for row in tqdm(input_data):
        tokens = row['tokens']
        step_input = {
            'tokens': [tokens]
        }
        step_output = {
            'tokens': [],
            'entities': [],
            'relations': [],
            'entity_gts': [],
            'relation_gts': [],
            'entities_': [],
            'relations_': [],
            'agg_probs': []
            # 'table_probs': []
        }
        rets = model.predict_step(step_input)
        rets = {k: list(v[0]) for k, v in rets.items() if k in kept_fields}
        # Append the gt
        for e in row['entities']:
            k = list(e)
            k.append(' '.join(tokens[e[0]: e[1]]))
            step_output['entity_gts'].append(k)
        for r in row['relations']:
            k = list(r)
            k.append(' '.join(tokens[r[0]: r[1]]))
            k.append(' '.join(tokens[r[2]: r[3]]))
            step_output['relation_gts'].append(k)

        # Append the predicted
        for e in rets['entity_preds']:
            k = list(e)
            # k.append(' '.join(tokens[e[0]: e[1]]))
            step_output['entities'].append(k)
        for r in rets['relation_preds']:
            k = list(r)
            # k.append(' '.join(tokens[r[0]: r[1]]))
            # k.append(' '.join(tokens[r[2]: r[3]]))
            step_output['relations'].append(k)

        # Predicted with descriptions
        table_probs = np.asarray(rets['table_probs'])
        for e in rets['entity_preds']:
            k = list(e)
            k.append(' '.join(tokens[e[0]: e[1]]))
            k.append(table_probs.diagonal()[e[0]: e[1]].tolist())
            step_output['entities_'].append(k)
        for r in rets['relation_preds']:
            k = list(r)
            k.append(' '.join(tokens[r[0]: r[1]]))
            k.append(' '.join(tokens[r[2]: r[3]]))
            k.append(table_probs[r[0]: r[1], r[2]: r[3]].tolist())
            step_output['relations_'].append(k)

        #step_output['table_probs'] = table_probs.tolist()
        step_output['agg_probs'] = np.asarray(table_probs.tolist()).min()

        step_output['tokens'] = tokens
        outputs.append(step_output)
    with open(output_path, 'w') as f:
        json.dump(outputs, f)


def load_model(path):
    # ********* MODIFY HERE *********
    args = {
        'model_read_ckpt': path,
        'lm_emb_path': 'albert-xxlarge-v1',  # language model name
        'pretrained_wv': '../wv/glove.6B.100d.conll04.txt',  # original GloVe embeddings
        'vocab_size': 400100,  # GloVe contains 400,000 words
        'device': 'cpu',
    }
    if args['device'] is not None and args['device'] != 'cpu':
        torch.cuda.set_device(args['device'])
    elif args['device'] is None:
        if torch.cuda.is_available():
            gpu_idx, gpu_mem = set_max_available_gpu()
            args['device'] = f"cuda:{gpu_idx}"
        else:
            args['device'] = "cpu"

    # Load the config file of trained ckpt
    with open(args['model_read_ckpt'] + '.json', 'r') as f:
        config = Config(**json.load(f))

        # load language model to dynamically calculate the contextualized word embeddings
        config.lm_emb_path = args['lm_emb_path']
        # assign device
        config.device = args['device']

    model = JointModel(config)
    model.load_ckpt(args['model_read_ckpt'])
    return model