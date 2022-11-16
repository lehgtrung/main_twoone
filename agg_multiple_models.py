
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


def evaluate_model(preds, gts):
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


def evaluate_multiple_models(inputs, model1, model2, model3):
    # feed each input to each model via predict_step and compute f1
    outputs = []
    for row in inputs:
        tokens = row['tokens']
        step_input = {
            'tokens': [tokens]
        }
        pred1 = model1.forward_step(step_input)
        ner_tag_logits1, re_tag_logits1 = pred1['ner_tag_logits'], pred1['re_tag_logits']

        pred2 = model2.forward_step(step_input)
        ner_tag_logits2, re_tag_logits2 = pred2['ner_tag_logits'], pred2['re_tag_logits']

        pred3 = model3.forward_step(step_input)
        ner_tag_logits3, re_tag_logits3 = pred3['ner_tag_logits'], pred3['re_tag_logits']

        ner_tag_logits = (ner_tag_logits1 + ner_tag_logits2 + ner_tag_logits3) / 3
        re_tag_logits = (re_tag_logits1 + re_tag_logits2 + re_tag_logits3) / 3

        entity_preds = process_ner_logits(model1, ner_tag_logits, pred1['masks'])
        relation_preds = process_re_logits(model1, re_tag_logits, entity_preds)

        prediction = model1.predict_step(step_input)
        print(prediction['entity_preds'])
        print(prediction['relation_preds'])

        output = {
            'tokens': step_input['tokens'],
            'entities': entity_preds[0],
            'relations': relation_preds[0]
        }
        print(output)
        exit()
        outputs.append(output)
    return outputs


if __name__ == '__main__':
    pred_path = 'datasets/core_conll04/train.conll04.json'
    gt_path = 'datasets/core_conll04/train.conll04.json'
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    with open(gt_path, 'r') as f:
        gts = json.load(f)

    model_path1 = 'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_0'
    model_path2 = 'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_1'
    model_path3 = 'datasets/methods/tri_training/conll04_30/fold=1/models/labeled_2'

    # evaluate_model(preds, gts)
    model1 = load_model(path=model_path1)
    model2 = load_model(path=model_path2)
    model3 = load_model(path=model_path3)

    print('FINISH LOADING MODELS')

    print(evaluate_multiple_models(gts, model1, model2, model3))



























