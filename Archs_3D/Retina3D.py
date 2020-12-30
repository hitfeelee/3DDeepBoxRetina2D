import torch.nn as nn
from nets.Backbone import BackBone
from Archs_2D.FPN import FPN
from nets.MobilenetV2 import InvertedResidual
from nets.ShapeSpec import ShapeSpec
from utils.FocalLoss import sigmoid_focal_loss_jit
from utils.SmoothLoss import smooth_l1_loss
import math
from collections import OrderedDict
from Archs_2D.BBox import *
from postprocess.nms import *
from postprocess.instance import *
from utils.ParamList import ParamList
import time
from torch.nn import functional as F
from Archs_3D.Retina3DGroundTruthCoder import GroundTruthCoder
from Archs_3D.MultiBin import MultiBin

def orientation_loss(pred_orient, gt_orient, gt_conf):

    batch_size, bins = gt_conf.size()
    indexes = torch.argmax(gt_conf, dim=1)
    indexes_cos = (indexes * bins).long()
    indexes_sin = (indexes * bins + 1).long()
    batch_ids = torch.arange(batch_size)
    # extract just the important bin

    theta_diff = torch.atan2(gt_orient[batch_ids, indexes_sin], gt_orient[batch_ids,indexes_cos])
    estimated_theta_diff = torch.atan2(pred_orient[batch_ids, indexes_sin], pred_orient[batch_ids,indexes_cos])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean() + 1

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_regress_to_N_HWA_K_and_concat(pred_labels,
                                                      bins=2,
                                                      num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    pred_classes = pred_labels["class"]
    pred_bbox_offsets = pred_labels["bbox_offset"]
    pred_dim_offsets = pred_labels["dim_offset"]
    pred_orient_offsets = pred_labels["orient_offset"]
    pred_bin_confs = pred_labels["bin_conf"]
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    pred_classes_flattened = [permute_to_N_HWA_K(x, num_classes) for x in pred_classes]
    pred_bbox_offsets_flattened = [permute_to_N_HWA_K(x, 4) for x in pred_bbox_offsets]
    pred_dim_offsets_flattened = [permute_to_N_HWA_K(x, 3) for x in pred_dim_offsets]
    pred_orient_offsets_flattened = [permute_to_N_HWA_K(x, bins*2) for x in pred_orient_offsets]
    pred_bin_confs_flattened = [permute_to_N_HWA_K(x, bins) for x in pred_bin_confs]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    pred_classes = torch.cat(pred_classes_flattened, dim=1).view(-1, num_classes)
    pred_bbox_offsets = torch.cat(pred_bbox_offsets_flattened, dim=1).view(-1, 4)
    pred_dim_offsets = torch.cat(pred_dim_offsets_flattened, dim=1).view(-1, 3)
    pred_orient_offsets = torch.cat(pred_orient_offsets_flattened, dim=1).view(-1, bins*2)
    pred_bin_confs = torch.cat(pred_bin_confs_flattened, dim=1).view(-1, bins)
    return {
            'class': pred_classes,
            'bbox_offset': pred_bbox_offsets,
            'dim_offset': pred_dim_offsets,
            'orient_offset': pred_orient_offsets,
            'bin_conf': pred_bin_confs
        }


class Retina3DArch(BackBone):
    @property
    def ground_truth_coder(self):
        return self._ground_truth_coder

    @property
    def bbox_coder(self):
        return self._bbox_coder

    def __init__(self, base_net: nn.Module, configs=None):
        """Compose a SSD model using the given components.
        """
        super(Retina3DArch, self).__init__(configs)
        print(configs)
        self.num_classes = configs.RETINANET.NUM_CLASSES
        self.focal_loss_alpha = configs.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = configs.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = configs.RETINANET.SMOOTH_L1_LOSS_BETA
        self.device = torch.device(configs.DEVICE if torch.cuda.is_available() else "cpu")
        self.in_size = configs.INTENSOR_SIZE
        # Inference parameters:
        self.score_threshold = configs.DETECTOR.SCORE_THRESH_TEST
        self.topk_candidates = configs.DETECTOR.TOPK_CANDIDATES_TEST
        self.nms_threshold = configs.DETECTOR.NMS_THRESH_TEST
        self.max_detections_per_image = configs.DETECTOR.DETECTIONS_PER_IMAGE
        self.bins = configs.MULTIBIN.BINS

        # build model net
        # ========base net======== #
        self.base_net = base_net
        # ========extra net======== #
        base_net_feature_shapes = self.base_net.OutShapeSpec
        extra_in_channels = base_net_feature_shapes[list(base_net_feature_shapes.keys())[-1]].channels
        extra_in_stride = base_net_feature_shapes[list(base_net_feature_shapes.keys())[-1]].stride
        self.extras = self._build_extra_nets(extra_in_channels, extra_in_stride)
        # ========fpn net======== #
        self.fpn = FPN(base_net_feature_shapes, configs.RETINANET.OUT_CHANNELS, configs)
        # ========anchor and bbox coder======== #
        od_feature_shapes = [self.OutShapeSpec[f] for f in configs.RETINANET.OD_FEATURES]

        multibin = MultiBin(configs.MULTIBIN.BINS, configs.MULTIBIN.OVERLAP, device=self.device)
        self._ground_truth_coder = GroundTruthCoder(configs, multibin, od_feature_shapes)
        num_anchor = self._ground_truth_coder.num_anchor
        # ========subnet headers net for classification and regression======== #
        self.classification_headers, self.bbox_regress_headers, \
        self.dim_regress_headers, self.orient_regress_headers, self.bin_conf_headers = self._build_subnet_header(num_anchor)
        self.add_module('base_net', self.base_net)
        self.add_module('extras', self.extras)
        self.add_module('fpn', self.fpn)
        self.add_module('classification_headers', self.classification_headers)
        self.add_module('bbox_regress_headers', self.bbox_regress_headers)
        self.add_module('dim_regress_headers', self.dim_regress_headers)
        self.add_module('orient_regress_headers', self.orient_regress_headers)
        self.add_module('bin_conf_headers', self.bin_conf_headers)
        self.init()

    def forward(self, x: torch.Tensor, gt_labels:dict=None,
                is_training:bool = False):
        confidences = []
        bboxes = []
        dims = []
        orients = []
        bin_confs = []
        t1 = time.time()
        x = self.base_net(x)
        t2 = time.time()
        # print('backbone inference time: ', t2 - t1)
        self.features = self.base_net.features
        self.fpn(self.features)
        self.features = OrderedDict(self.features, **self.fpn.features)
        for i, layer in enumerate(self.extras):
            x = layer(x)
            self.features['extra_layer{}'.format(i)] = x

        for feature_key in self.configs.RETINANET.OD_FEATURES:
            confidences.append(self.classification_headers(self.features[feature_key]))
            bboxes.append(self.bbox_regress_headers(self.features[feature_key]))
            dims.append(self.dim_regress_headers(self.features[feature_key]))
            orients_logits = self.orient_regress_headers(self.features[feature_key])
            N, C, H, W = orients_logits.size()
            orients_logits = orients_logits.view(N, -1, 2, H, W) # [N, (A X bins), 2, H, W]
            orients_logits = F.normalize(orients_logits, dim=2)
            orients.append(orients_logits.view(N, -1, H, W))
            bin_confs.append(self.bin_conf_headers(self.features[feature_key]))

        pred_labels = {
            'class': confidences,
            'bbox_offset': bboxes,
            'dim_offset': dims,
            'orient_offset': orients,
            'bin_conf': bin_confs
        }
        if is_training:
            losses = self.losses(pred_labels, gt_labels)
            return losses
        else:
            batch_size = x.size(0)
            anchors = self._ground_truth_coder.gen_anchors(batch_size)
            results = self.inference(pred_labels, anchors)
            return results

    def _build_extra_nets(self, in_channels, in_stride):
        extras = []
        self._extra_feature_channels = []
        self._extra_feature_strides = []
        extra = [nn.Conv2d(in_channels, in_channels, 3, dilation=6, padding=6),
                 nn.Dropout(0.5),
                 nn.Conv2d(in_channels, in_channels, 1, dilation=6, padding=0),
                 nn.Dropout(0.5)
                 ]
        extras.append(nn.Sequential(*extra))
        self._extra_feature_channels.append(in_channels)
        self._extra_feature_strides.append(in_stride)

        if self.configs.EXTRANET.USE_INV_RES:
            for _ in range(self.configs.EXTRANET.NUMLAYERS):
                extra = [InvertedResidual(in_channels if i == 0 else self.configs.RETINANET.OUT_CHANNELS,
                                          self.configs.RETINANET.OUT_CHANNELS, 2 if i == 0 else 1,
                                          expand_ratio=6, use_batch_norm=True)
                         for i in range(self.configs.EXTRANET.NUMCONVS)]
                extra = nn.Sequential(*extra)
                extras.append(extra)
                in_channels = self.configs.RETINANET.OUT_CHANNELS
        else:
            for _ in range(self.configs.EXTRANET.NUMLAYERS):
                extra = [nn.Sequential(nn.Conv2d(in_channels if i == 0 else self.configs.RETINANET.OUT_CHANNELS,
                                                 self.configs.RETINANET.OUT_CHANNELS, 1, 1, padding=0),
                                       nn.BatchNorm2d(self.configs.RETINANET.OUT_CHANNELS),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS,
                                                 3, 2 if i == 0 else 1, padding=1),
                                       nn.BatchNorm2d(self.configs.RETINANET.OUT_CHANNELS),
                                       nn.ReLU(inplace=True))
                         for i in range(self.configs.EXTRANET.NUMCONVS)]
                extra = nn.Sequential(*extra)
                extras.append(extra)
                in_channels = self.configs.RETINANET.OUT_CHANNELS
        self._extra_feature_strides = self._extra_feature_strides + [in_stride * (2 ** (i + 1))
                                                                     for i in range(self.configs.EXTRANET.NUMLAYERS)]
        self._extra_feature_channels = self._extra_feature_channels + \
                                       [self.configs.RETINANET.OUT_CHANNELS] * self.configs.EXTRANET.NUMLAYERS
        return nn.Sequential(*extras)

    def _build_subnet_header(self, num_anchor):

        classifications = [nn.Sequential(
            nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))
            for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
        classifications.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.num_classes * num_anchor,
                                         3, stride=1, padding=1))
        bbox_regress = [nn.Sequential(
            nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))
            for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
        bbox_regress.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, 4 * num_anchor,
                                      3, stride=1, padding=1))

        dim_regress = [nn.Sequential(
            nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
            for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
        dim_regress.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, 3 * num_anchor,
                                     3, stride=1, padding=1))

        orient_regress = [nn.Sequential(
            nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
            for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
        orient_regress.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.bins*2 * num_anchor,
                                        3, stride=1, padding=1))

        bin_conf = [nn.Sequential(
            nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True))
            for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
        bin_conf.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.bins * num_anchor,
                                  3, stride=1, padding=1))

        classification_headers = nn.Sequential(*classifications)
        bbox_regress = nn.Sequential(*bbox_regress)
        dim_regress = nn.Sequential(*dim_regress)
        orient_regress = nn.Sequential(*orient_regress)
        bin_conf = nn.Sequential(*bin_conf)

        return classification_headers, bbox_regress, dim_regress, orient_regress, bin_conf

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.fpn.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_header_init_)
        self.bbox_regress_headers.apply(_header_init_)
        self.dim_regress_headers.apply(_header_init_)
        self.orient_regress_headers.apply(_header_init_)
        self.bin_conf_headers.apply(_header_init_)

        # Use prior in model initialization to improve stability
        self.classification_headers[-1].apply(self._classification_header_init_)

    @property
    def OutShapeSpec(self):
        extra_shapes = OrderedDict()

        for i, s in enumerate(self._extra_feature_strides):
            extra_shapes['extra_layer{}'.format(i)] = ShapeSpec(channels=self._extra_feature_channels[i],
                                                                stride=s)
        feature_shapes = OrderedDict(self.base_net.OutShapeSpec, **self.fpn.OutShapeSpec)
        feature_shapes = OrderedDict(feature_shapes, **extra_shapes)

        return feature_shapes

    def losses(self, pred_labels, gt_labels):

        pred_labels = permute_all_cls_and_regress_to_N_HWA_K_and_concat(pred_labels, self.bins, self.num_classes)
        pred_classes = pred_labels["class"]
        pred_bbox_offsets = pred_labels["bbox_offset"]
        pred_dim_offsets = pred_labels["dim_offset"]
        pred_orient_offsets = pred_labels["orient_offset"]
        pred_bin_confs = pred_labels["bin_conf"]

        targets = self.ground_truth_coder.encode(gt_labels, device=pred_classes.device)
        gt_classes = targets["class"]
        gt_bbox_offsets = targets["bbox_offset"]
        gt_dim_offsets = targets["dim_offset"]
        gt_orients = targets["orient_offset"]
        gt_bin_confs = targets["bin_conf"]

        gt_classes = gt_classes.flatten().long()
        gt_bbox_offsets = gt_bbox_offsets.view(-1, 4)
        gt_dim_offsets = gt_dim_offsets.view(-1, 3)
        gt_orients = gt_orients.view(-1, self.bins*2)
        gt_bin_confs = gt_bin_confs.view(-1, self.bins)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_classes)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_classes[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground) * 0.

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_bbox_offsets[foreground_idxs],
            gt_bbox_offsets[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground) * 0.

        orient_loss = orientation_loss(pred_orient_offsets[foreground_idxs],
                                       gt_orients[foreground_idxs],
                                       gt_bin_confs[foreground_idxs]) * 1.
        dim_loss = smooth_l1_loss(pred_dim_offsets[foreground_idxs],
                                  gt_dim_offsets[foreground_idxs],
                                  beta=self.smooth_l1_loss_beta,
                                  reduction="sum")/max(1, num_foreground) * 1.

        gt_conf = torch.argmax(gt_bin_confs[foreground_idxs], dim=1)
        bin_conf_loss = F.cross_entropy(pred_bin_confs[foreground_idxs],
                                        gt_conf,
                                        reduction="sum"
                                        )/ max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg,
                "loss_dim": dim_loss,
                "loss_orient": orient_loss,
                "loss_bin_conf": bin_conf_loss}

    def inference(self, pred_labels, anchors):
        """
        Arguments:
            pred_labels: type dict
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        pred_classes = pred_labels["class"]
        pred_bbox_offsets = pred_labels["bbox_offset"]
        pred_dim_offsets = pred_labels["dim_offset"]
        pred_orient_offsets = pred_labels["orient_offset"]
        pred_bin_confs = pred_labels["bin_conf"]
        pred_classes = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_classes]
        pred_bbox_offsets = [permute_to_N_HWA_K(x, 4) for x in pred_bbox_offsets]
        pred_dim_offsets = [permute_to_N_HWA_K(x, 3) for x in pred_dim_offsets]
        pred_orient_offsets = [permute_to_N_HWA_K(x, self.bins*2) for x in pred_orient_offsets]
        pred_bin_confs = [permute_to_N_HWA_K(x, self.bins) for x in pred_bin_confs]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)
        pred_labels_per_image = dict()
        for img_idx, anchors_per_image in enumerate(anchors):
            pred_labels_per_image['class'] = [pred_classes_per_level[img_idx] for pred_classes_per_level in pred_classes]
            pred_labels_per_image['bbox_offset'] = [pred_bbox_offsets_per_level[img_idx] for pred_bbox_offsets_per_level in pred_bbox_offsets]
            pred_labels_per_image['dim_offset'] = [pred_dim_offsets_per_level[img_idx] for pred_dim_offsets_per_level in pred_dim_offsets]
            pred_labels_per_image['orient_offset'] = [pred_orient_offsets_per_level[img_idx] for pred_orient_offsets_per_level in pred_orient_offsets]
            pred_labels_per_image['bin_conf'] = [pred_bin_confs_per_level[img_idx] for pred_bin_confs_per_level in pred_bin_confs]

            results_per_image = self.inference_single_image(pred_labels_per_image, anchors_per_image)
            results.append(results_per_image)
        return results

    def inference_single_image(self, pred_labels, anchors):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred_classes = pred_labels["class"]
        pred_bbox_offsets = pred_labels["bbox_offset"]
        pred_dim_offsets = pred_labels["dim_offset"]
        pred_orient_offsets = pred_labels["orient_offset"]
        pred_bin_confs = pred_labels["bin_conf"]
        pred_classes_all = []
        pred_scores_all = []
        pred_bboxes_all = []
        pred_dims_all = []
        pred_orients_all = []

        # Iterate over every feature level
        for i, anchors_i in enumerate(anchors):
            pred_classes_i = pred_classes[i]
            pred_bbox_offsets_i = pred_bbox_offsets[i]
            pred_dim_offsets_i = pred_dim_offsets[i]
            pred_orient_offsets_i = pred_orient_offsets[i]
            pred_bin_confs_i = pred_bin_confs[i]
            # (HxWxAxK,)
            pred_classes_i = pred_classes_i.flatten().sigmoid()
            anchors_i = anchors_i.to(pred_classes_i.device)
            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, pred_bbox_offsets_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            pred_scores_i, topk_idxs = pred_classes_i.sort(descending=True)
            pred_scores_i = pred_scores_i[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = pred_scores_i > self.score_threshold
            if keep_idxs.float().sum() <= 0:
                continue
            pred_scores_i = pred_scores_i[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            pred_bbox_offsets_i = pred_bbox_offsets_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            pred_dim_offsets_i = pred_dim_offsets_i[anchor_idxs]
            pred_orient_offsets_i = pred_orient_offsets_i[anchor_idxs]
            pred_bin_confs_i = pred_bin_confs_i[anchor_idxs]
            # predict boxes
            pred_bbox_i = self._ground_truth_coder.decode_bbox(pred_bbox_offsets_i, anchors_i.tensor)
            # predict dimensions
            pred_dims_i = self._ground_truth_coder.decode_dimension(pred_dim_offsets_i, classes_idxs)
            # predict orientation
            pred_orients_i = self._ground_truth_coder.decode_orient(pred_orient_offsets_i, pred_bin_confs_i)
            pred_bboxes_all.append(pred_bbox_i)
            pred_scores_all.append(pred_scores_i)
            pred_classes_all.append(classes_idxs)
            pred_dims_all.append(pred_dims_i)
            pred_orients_all.append(pred_orients_i)

        result = ParamList(self.in_size, is_train=False)
        if len(pred_bboxes_all) > 0:
            pred_bboxes_all, pred_scores_all, pred_classes_all, pred_dims_all, pred_orients_all = [
                torch.cat(x) for x in [pred_bboxes_all, pred_scores_all, pred_classes_all, pred_dims_all, pred_orients_all]
            ]
            keep = batched_nms(pred_bboxes_all, pred_scores_all, pred_classes_all, self.nms_threshold)
            keep = keep[: self.max_detections_per_image]
            result.add_field("class", pred_classes_all[keep])
            result.add_field("score", pred_scores_all[keep])
            result.add_field("bbox", pred_bboxes_all[keep])
            result.add_field("dimension", pred_dims_all[keep])
            result.add_field("orientation", pred_orients_all[keep])
        else:
            result.add_field("class", pred_classes_all)
            result.add_field("score", pred_scores_all)
            result.add_field("bbox", pred_bboxes_all)
            result.add_field("dimension", pred_dims_all)
            result.add_field("orientation", pred_orients_all)
        return result

    def _classification_header_init_(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            bias_value = -math.log((1 - self.configs.RETINANET.PRIOR_PROB) / self.configs.RETINANET.PRIOR_PROB)
            nn.init.constant_(m.bias, bias_value)

def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def _header_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)
