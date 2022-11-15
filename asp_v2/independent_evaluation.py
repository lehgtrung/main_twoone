
import json
import torch


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


if __name__ == '__main__':
    pred_path = 'datasets/core_conll04/train.CoNLL04.json'
    gt_path = 'datasets/core_conll04/train.CoNLL04.json'
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    with open(gt_path, 'r') as f:
        gts = json.load(f)

    evaluate_model(preds, gts)

























