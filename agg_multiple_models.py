
import json
import torch
import numpy as np
from prediction_helper import load_model


def get_metrics(sent_list, preds_list, labels_list):
    n_correct, n_pred, n_label = 0, 0, 0
    i_count = 0
    for sent, preds, labels in zip(sent_list, preds_list, labels_list):
        preds = set(preds)
        labels = {tuple(x) for x in labels}

        n_pred += len(preds)
        n_label += len(labels)
        n_correct += len(preds & labels)

        i_count += 1

    precision = n_correct / (n_pred + 1e-8)
    recall = n_correct / (n_label + 1e-8)
    f1 = 2 / (1 / (precision + 1e-8) + 1 / (recall + 1e-8) + 1e-8)

    return precision, recall, f1


def evaluate_model(preds, gts, logger=None):
    with torch.no_grad():
        sents = []
        pred_entities = []
        pred_relations = []
        pred_relations_wNER = []
        label_entities = []
        label_relations = []
        label_relations_wNER = []
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            pred['entities'] = [[tuple(e) for e in pred['entities']]]
            gt['entities'] = [gt['entities']]
            pred['relations'] = [set(tuple(e) for e in pred['relations'])]
            gt['relations'] = [gt['relations']]
            pred_span_to_etype = [{(ib, ie): etype for ib, ie, etype in x} for x in pred['entities']]
            label_span_to_etype = [{(ib, ie): etype for ib, ie, etype in x} for x in gt['entities']]
            pred_entities += pred['entities']
            label_entities += gt['entities']
            pred_relations += pred['relations']
            label_relations += gt['relations']

            pred_relations_wNER += [
                [
                    (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
                ] for x, m in zip(pred['relations'], pred_span_to_etype)
            ]
            label_relations_wNER += [
                [
                    (ib, ie, m[(ib, ie)], jb, je, m[(jb, je)], rtype) for ib, ie, jb, je, rtype in x
                ] for x, m in zip(gt['relations'], label_span_to_etype)
            ]

            sents += [pred['tokens']]

        rets = {}
        rets['entity_p'], rets['entity_r'], rets['entity_f1'] = get_metrics(
            sents, pred_entities, label_entities)
        rets['relation_p'], rets['relation_r'], rets['relation_f1'] = get_metrics(
            sents, pred_relations, label_relations)
        rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER'] = get_metrics(
            sents, pred_relations_wNER, label_relations_wNER)

    precision, recall, f1 = rets['entity_p'], rets['entity_r'], rets['entity_f1']
    print(f">> entity prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
    precision, recall, f1 = rets['relation_p'], rets['relation_r'], rets['relation_f1']
    print(f">> relation prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
    precision, recall, f1 = rets['relation_p_wNER'], rets['relation_r_wNER'], rets['relation_f1_wNER']
    print(f">> relation with NER prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")

    if logger:
        logger.info(f">> entity prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        logger.info(f">> relation prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")
        logger.info(f">> relation with NER prec:{precision:.4f}, rec:{recall:.4f}, f1:{f1:.4f}")


def process_ner_logits(model, ner_tag_logits, mask):
    mask_np = mask.cpu().detach().numpy()
    if model.config.crf == 'CRF':
        ner_tag_preds = model.crf_layer.decode(ner_tag_logits, mask=mask)
    elif not model.config.crf:
        ner_tag_preds = ner_tag_logits.argmax(dim=-1).cpu().detach().numpy()
    else:
        raise Exception('not a compatible decode')

    ner_tag_preds = np.array(ner_tag_preds)
    ner_tag_preds *= mask_np
    ner_tag_preds = model.ner_tag_indexing.inv(ner_tag_preds)
    entity_preds = model._postprocess_entities(ner_tag_preds)
    return entity_preds


def process_re_logits(model, re_tag_logits, entity_preds):
    relation_preds = model._postprocess_relations(re_tag_logits, entity_preds)
    return relation_preds


def aggregate_multiple_models(inputs, models):
    # feed each input to each model via predict_step and compute f1
    outputs = []
    n = len(models)
    for model in models:
        model.eval()
    for row in inputs:
        tokens = row['tokens']
        step_input = {
            'tokens': [tokens]
        }

        lst_ner_tag_logits = []
        lst_re_tag_logits = []

        for model in models:
            _pred = model.forward_step(step_input)
            _ner_tag_logits, _re_tag_logits = _pred['ner_tag_logits'], _pred['re_tag_logits']
            lst_ner_tag_logits.append(_ner_tag_logits)
            lst_re_tag_logits.append(_re_tag_logits)

        ner_tag_logits = torch.mean(torch.stack(lst_ner_tag_logits), dim=0)
        re_tag_logits = torch.mean(torch.stack(lst_re_tag_logits), dim=0)

        entity_preds = process_ner_logits(models[0], ner_tag_logits, _pred['masks'])
        relation_preds = process_re_logits(models[0], re_tag_logits, entity_preds)

        output = {
            'tokens': step_input['tokens'],
            'entities': entity_preds[0],
            'relations': relation_preds[0]
        }
        print(output)

        for i in range(len(models)):
            print(f'Model {i} predicts')
            pred0 = models[i].predict_step(step_input)
            print(pred0['entity_preds'])
            print(pred0['relation_preds'])
        print('======================')
        outputs.append(output)
    return outputs


def evaluate_multiple_models(eval_path,
                             test_path,
                             model_paths,
                             logger=None):
    with open(eval_path, 'r') as f:
        eval_set = json.load(f)
    with open(test_path, 'r') as f:
        test_set = json.load(f)
    models = []
    for path in model_paths:
        model = load_model(path=path)
        models.append(model)

    eval_outputs = aggregate_multiple_models(eval_set, models)
    test_outputs = aggregate_multiple_models(test_set, models)
    if logger:
        logger.info('Eval results')
    evaluate_model(eval_outputs, eval_set, logger=logger)
    if logger:
        logger.info('Test results')
    evaluate_model(test_outputs, test_set, logger=logger)


if __name__ == '__main__':
    # pred_path = 'datasets/core_conll04/temp.conll04.json'
    # gt_path = 'datasets/core_conll04/test.conll04.json'
    # with open(pred_path, 'r') as f:
    #     preds = json.load(f)
    #
    # with open(gt_path, 'r') as f:
    #     gts = json.load(f)

    model_paths = [
        'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_0',
        'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_1',
        'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_2'
    ]
    DEFAULT_TEST_PATH = './datasets/core_conll04/test.conll04.json'
    DEFAULT_VALID_PATH = './datasets/core_conll04/valid.conll04.json'

    evaluate_multiple_models(DEFAULT_VALID_PATH,
                             DEFAULT_TEST_PATH,
                             model_paths)

    # evaluate_model(preds, gts, logger=None)























