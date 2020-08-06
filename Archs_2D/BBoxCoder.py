
from Archs_2D.AnchorGenerator import AnchorGenerator
from Archs_2D.BBox import *
from Archs_2D.Matcher import *
import torch
import copy

class BBoxCoder(object):
    def __init__(self, configs, od_feature_shape):
        self.configs = configs
        self.num_classes = configs.RETINANET.NUM_CLASSES
        self.od_feature_shape = od_feature_shape
        self.anchor_gener = AnchorGenerator(configs, od_feature_shape)
        self.anchors = self.anchor_gener(configs.INTENSOR_SHAPE)
        self.box2box_transform = Box2BoxTransform(weights=configs.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            configs.RETINANET.IOU_THRESHOLDS,
            configs.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

    def gen_anchors(self, batch_size):
        return [copy.deepcopy(self.anchors) for _ in range(batch_size)]

    @property
    def num_anchor(self):
        return self.anchor_gener.num_cell_anchors[0]

    @torch.no_grad()
    def encode(self, labels:list, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        device = torch.device(device)
        batch_size = len(labels)
        anchors = self.gen_anchors(batch_size)
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [BBoxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, labels_per_image in zip(anchors, copy.deepcopy(labels)):
            anchors_per_image = anchors_per_image.to(device)
            labels_per_image.gt_bboxes = labels_per_image.gt_bboxes.to(device)
            labels_per_image.gt_classes = labels_per_image.gt_classes.to(device)
            match_quality_matrix = pairwise_iou(labels_per_image.gt_bboxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(labels_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_bboxes = labels_per_image.gt_bboxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_bboxes.tensor
                )

                gt_classes_i = labels_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    @torch.no_grad()
    def decode(self, delta_bboxes, bboxes):
        return self.box2box_transform.apply_deltas(delta_bboxes, bboxes)
