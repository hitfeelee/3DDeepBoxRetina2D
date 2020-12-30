


from Archs_2D.AnchorGenerator import AnchorGenerator
from Archs_2D.BBox import *
from Archs_2D.Matcher import *
import torch
import copy
import numpy as np

class GroundTruthCoder(object):
    def __init__(self, configs, multibin, od_feature_shapes):
        super(GroundTruthCoder, self).__init__()
        self.multibin = multibin
        self.configs = configs
        self.num_classes = configs.RETINANET.NUM_CLASSES
        self.dim_def = torch.tensor(configs.DATASET.DIM_MEAN,dtype=torch.float32)
        self.od_feature_shapes = od_feature_shapes
        self.anchor_gener = AnchorGenerator(configs, od_feature_shapes)
        self.anchors = self.anchor_gener(configs.INTENSOR_SIZE)
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
    def encode(self, targets=None, device=torch.device('cpu')):

        device = torch.device(device)
        batch_size = len(targets)
        anchors = self.gen_anchors(batch_size)
        gt_classes = []
        gt_bbox_offsets = []
        gt_dims_offsets = []
        gt_orients = []
        gt_bin_confs = []
        anchors = [BBoxes.cat(bbox_i) for bbox_i in anchors]
        for bboxes_per_image, labels_per_image in zip(anchors, targets):
            bboxes_per_image = bboxes_per_image.to(device)
            gt_bboxes_per_image = labels_per_image.get_field('bbox').to(device)
            gt_classes_per_image = labels_per_image.get_field('class').to(device)
            masks_per_image = labels_per_image.get_field('mask').to(device)
            match_quality_matrix = pairwise_iou(gt_bboxes_per_image, bboxes_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = masks_per_image.sum() > 0
            if has_gt:
                matched_masks = masks_per_image[gt_matched_idxs].bool().bitwise_not()
                # ground truth box regression
                matched_gt_bboxes = gt_bboxes_per_image[gt_matched_idxs]
                matched_gt_bboxes = self.box2box_transform.get_deltas(
                    bboxes_per_image.tensor, matched_gt_bboxes.tensor
                )

                matched_gt_classes = gt_classes_per_image[gt_matched_idxs]
                if labels_per_image.has_field('dimension'):
                    matched_gt_dims = self.encode_dimension(
                        labels_per_image.get_field('dimension')[gt_matched_idxs],
                        matched_gt_classes,
                        device=device)
                    gt_dims_offsets.append(matched_gt_dims)
                if labels_per_image.has_field('alpha'):
                    matched_orients, matched_bin_confs = self.encode_orient(
                        labels_per_image.get_field('alpha')[gt_matched_idxs],
                        device=device)
                    gt_orients.append(matched_orients)
                    gt_bin_confs.append(matched_bin_confs)

                # Anchors with label 0 are treated as background.
                matched_gt_classes[matched_masks] = self.num_classes
                matched_gt_classes[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                matched_gt_classes[anchor_labels == -1] = -1

            else:
                num = gt_matched_idxs.size()[0]
                matched_gt_classes = torch.zeros_like(gt_matched_idxs, dtype=torch.float32) + self.num_classes
                matched_gt_bboxes = torch.zeros_like(bboxes_per_image.tensor)
                if labels_per_image.has_field('dimension'):
                    matched_gt_dims = torch.zeros((num, 3), dtype=torch.float32, device=device)
                    gt_dims_offsets.append(matched_gt_dims)

                if labels_per_image.has_field('alpha'):
                    matched_orients = torch.zeros((num, self.multibin.bin_num*2),
                                                       dtype=torch.float32, device=device)
                    matched_bin_confs = torch.zeros((num, self.multibin.bin_num),
                                                         dtype=torch.long, device=device)
                    gt_orients.append(matched_orients)
                    gt_bin_confs.append(matched_bin_confs)

            gt_classes.append(matched_gt_classes)
            gt_bbox_offsets.append(matched_gt_bboxes)


        return {
            'class': torch.stack(gt_classes),
            'bbox_offset': torch.stack(gt_bbox_offsets),
            'dim_offset': torch.stack(gt_dims_offsets) if len(gt_dims_offsets) else None,
            'orient_offset': torch.stack(gt_orients) if len(gt_orients) else None,
            'bin_conf': torch.stack(gt_bin_confs) if len(gt_bin_confs) else None
        }

    @torch.no_grad()
    def encode_orient(self, gt_alphas, device=torch.device('cpu')):

        num = list(gt_alphas.size())[0]
        # alpha is [-pi..pi], shift it to be [0..2pi]
        Orientation = torch.zeros((num, self.multibin.bin_num * 2), dtype=torch.float32, device=device)
        Confidence = torch.zeros((num, self.multibin.bin_num,), dtype=torch.long, device=device)
        alphas = gt_alphas + np.pi
        alphas = alphas.to(device)
        bin_idxs = self.multibin.get_bins(alphas)
        bin_ben_angles = self.multibin.get_bins_bench_angle(bin_idxs[1])
        angle_diff = alphas[bin_idxs[0]] - bin_ben_angles
        Confidence[bin_idxs] = 1
        Orientation[bin_idxs[0], bin_idxs[1]*self.multibin.bin_num] = torch.cos(angle_diff).to(torch.float32)
        Orientation[bin_idxs[0], bin_idxs[1]*self.multibin.bin_num + 1] = torch.sin(angle_diff).to(torch.float32)
        return Orientation, Confidence

    @torch.no_grad()
    def decode_orient(self, pred_alphas, pred_bin_confs):
        batch_size, bins = pred_bin_confs.size()
        argmax = torch.argmax(pred_bin_confs, dim=1)
        indexes_cos = (argmax * bins).long()
        indexes_sin = (argmax * bins + 1).long()
        batch_ids = torch.arange(batch_size).to(pred_bin_confs.device)
        # extract just the important bin
        alpha = torch.atan2(pred_alphas[batch_ids, indexes_sin], pred_alphas[batch_ids, indexes_cos])
        alpha += self.multibin.get_bin_bench_angle(argmax)
        # alpha is [0..2pi], shift it to be [-pi..pi]
        alpha -= np.pi
        i_pos = alpha > np.pi
        i_neg = alpha < -np.pi
        alpha[i_pos] -= 2*np.pi
        alpha[i_neg] += 2*np.pi
        return alpha

    @torch.no_grad()
    def encode_dimension(self, gt_dimensions, gt_classes, device=torch.device('cpu')):
        self.dim_def = self.dim_def.to(device)
        gt_dimensions = gt_dimensions.to(device)
        dim_defs = self.dim_def[gt_classes.long()]

        return gt_dimensions - dim_defs

    @torch.no_grad()
    def decode_dimension(self, pred_dimension_offsets, pred_classes):
        self.dim_def = self.dim_def.to(pred_classes.device)
        dim_defs = self.dim_def[pred_classes.long()]

        return pred_dimension_offsets + dim_defs

    @torch.no_grad()
    def decode_bbox(self, bbox_offsets, bboxes):

        return self.box2box_transform.apply_deltas(bbox_offsets, bboxes)
