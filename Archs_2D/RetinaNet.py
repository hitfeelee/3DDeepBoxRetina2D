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
from Archs_2D.BBoxCoder import BBoxCoder
import time

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


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = torch.cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


class RetinaNet(BackBone):
    @property
    def bbox_coder(self):
        return self._bbox_coder

    def __init__(self, base_net: nn.Module, configs=None):
        """Compose a SSD model using the given components.
        """
        super(RetinaNet, self).__init__(configs)
        print(configs)
        self.num_classes = configs.RETINANET.NUM_CLASSES
        self.focal_loss_alpha = configs.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = configs.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = configs.RETINANET.SMOOTH_L1_LOSS_BETA
        self.device = torch.device(configs.DEVICE if torch.cuda.is_available() else "cpu")
        self.in_shape = configs.INTENSOR_SHAPE
        # Inference parameters:
        self.score_threshold = configs.DETECTOR.SCORE_THRESH_TEST
        self.topk_candidates = configs.DETECTOR.TOPK_CANDIDATES_TEST
        self.nms_threshold = configs.DETECTOR.NMS_THRESH_TEST
        self.max_detections_per_image = configs.DETECTOR.DETECTIONS_PER_IMAGE
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
        self._bbox_coder = BBoxCoder(configs, od_feature_shapes)
        num_anchor = self.bbox_coder.num_anchor
        # ========subnet headers net for classification and regression======== #
        self.classification_headers, self.regression_headers = self._build_subnet_header(num_anchor)
        self.add_module('base_net', self.base_net)
        self.add_module('extras', self.extras)
        self.add_module('fpn', self.fpn)
        self.add_module('classification_headers', self.classification_headers)
        self.add_module('regression_headers', self.regression_headers)

    def forward(self, x: torch.Tensor,
                gt_classes:torch.Tensor = None, gt_bboxes:torch.Tensor=None,
                is_training:bool = False):
        confidences = []
        locations = []
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
            locations.append(self.regression_headers(self.features[feature_key]))

        if is_training:
            losses = self.losses(gt_classes, gt_bboxes, confidences, locations)
            return losses 
        else:
            batch_size = list(x.size())[0]
            anchors = self.bbox_coder.gen_anchors(batch_size)
            results = self.inference(confidences, locations, anchors)
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
        if self.configs.EXTRANET.USE_INV_RES:
            classifications = [InvertedResidual(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS,
                                                1, expand_ratio=6, use_batch_norm=True, use_bias=True)
                               for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
            classifications.append(InvertedResidual(self.configs.RETINANET.OUT_CHANNELS, self.num_classes * num_anchor,
                                                    1, expand_ratio=6, use_batch_norm=True, use_bias=True))
            regressions = [InvertedResidual(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS,
                                            1, expand_ratio=6, use_batch_norm=True, use_bias=True)
                           for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
            regressions.append(InvertedResidual(self.configs.RETINANET.OUT_CHANNELS, 4 * num_anchor,
                                                1, expand_ratio=6, use_batch_norm=True, use_bias=True))
        else:
            classifications = [nn.Sequential(
                nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1, padding=1),
                nn.ReLU(inplace=True))
                for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
            classifications.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.num_classes * num_anchor,
                                                    3, stride=1, padding=1))
            regressions = [nn.Sequential(
                nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, self.configs.RETINANET.OUT_CHANNELS, 3, stride=1, padding=1),
                nn.ReLU(inplace=True))
                for _ in range(self.configs.RETINANET.HEADER_NUMCONVS)]
            regressions.append(nn.Conv2d(self.configs.RETINANET.OUT_CHANNELS, 4 * num_anchor,
                                                3, stride=1, padding=1))
        classification_headers = nn.Sequential(*classifications)
        regression_headers = nn.Sequential(*regressions)

        return classification_headers, regression_headers
    @property
    def OutShapeSpec(self):
        extra_shapes = OrderedDict()

        for i, s in enumerate(self._extra_feature_strides):
            extra_shapes['extra_layer{}'.format(i)] = ShapeSpec(channels=self._extra_feature_channels[i],
                                                  stride=s)
        feature_shapes = OrderedDict(self.base_net.OutShapeSpec, **self.fpn.OutShapeSpec)
        feature_shapes = OrderedDict(feature_shapes, **extra_shapes)

        return feature_shapes

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def inference(self, box_cls, box_delta, anchors):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors):
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
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid()
            anchors_i = anchors_i.to(box_cls_i.device)
            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.bbox_coder.decode(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            torch.cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(self.in_shape)
        result.pred_boxes = BBoxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=False)
        self.fpn.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_header_init_)
        self.regression_headers.apply(_header_init_)
        # Use prior in model initialization to improve stability
        self.classification_headers[-1].apply(self._classification_header_init_)

    def init(self):
        # self.base_net.apply(_xavier_init_)
        self.fpn.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_header_init_)
        self.regression_headers.apply(_header_init_)

        # Use prior in model initialization to improve stability
        self.classification_headers[-1].apply(self._classification_header_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

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
