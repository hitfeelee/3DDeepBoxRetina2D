import torch
from Archs_2D.BBox import pairwise_iou

def batch_eval_PR(output, b_labels, threshold=0.5, device=torch.device('cpu')):
    b_tp, b_fp, b_fn = eval_tp_fp_fn(output, b_labels, threshold, device)
    precision = b_tp/max((b_tp + b_fp), 1)
    recall = b_tp/max((b_tp + b_fn), 1)
    return precision, recall


def eval_tp_fp_fn(output, b_labels, threshold=0.5, device=torch.device('cpu')):
    b_tp = 0
    b_fp = 0
    b_fn = 0
    for preds, labels in zip(output, b_labels):
        if len(preds) > 0:
            pred_bboxes = preds.get('pred_boxes')
            pred_classes = preds.get('pred_classes')
            gt_classes = labels.gt_classes.to(device)
            gt_bboxes = labels.gt_bboxes.to(device)
            match_iou = pairwise_iou(gt_bboxes, pred_bboxes)
            match_label, match_index = torch.max(match_iou, dim=0)
            keep_match = match_label > threshold
            match_index = match_index[keep_match]
            tp = torch.sum((gt_classes[match_index] == pred_classes[keep_match]).to(torch.float32))
            fp = torch.sum((gt_classes[match_index] != pred_classes[keep_match]).to(torch.float32))
            fn = (list(gt_classes.size())[0] - tp)
            b_tp += tp
            b_fp += fp
            b_fn += fn
        else:
            b_fn += (list(labels.gt_classes.size())[0])
    return b_tp, b_fp, b_fn
