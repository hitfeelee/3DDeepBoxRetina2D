import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nets.Backbone import BackBone
from nets.MobilenetV2 import InvertedResidual
from utils.Utils import *
from collections import OrderedDict
from nets.ShapeSpec import ShapeSpec
from Archs_3D.BinCoder import BinCoder
from Archs_3D.MultiBin import MultiBin

def OrientationLoss(pred_orient, gt_orient, gt_conf):

    batch_size = pred_orient.size()[0]
    indexes = torch.max(gt_conf, dim=1)[1]

    # extract just the important bin
    gt_orient = gt_orient[torch.arange(batch_size), indexes]
    pred_orient = pred_orient[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(gt_orient[:,1], gt_orient[:,0])
    estimated_theta_diff = torch.atan2(pred_orient[:,1], pred_orient[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean() + 1

class DeepBox3DArch(BackBone):
    @property
    def binCoder(self):
        return self._binCoder

    def __init__(self, base_net:nn.Module, configs):
        super(DeepBox3DArch, self).__init__(configs)
        self.alpha = configs.DEEPBOX3D.ALPHA
        self.w = configs.DEEPBOX3D.W
        self.device = torch.device(configs.DEVICE if torch.cuda.is_available() else "cpu")
        self.base_net = base_net
        self.bins = configs.DEEPBOX3D.BINS
        # ========base net======== #
        self.base_net = base_net
        # ========extra net======== #
        base_net_feature_shapes = self.base_net.OutShapeSpec
        extra_in_channels = base_net_feature_shapes[list(base_net_feature_shapes.keys())[-1]].channels
        extra_in_stride = base_net_feature_shapes[list(base_net_feature_shapes.keys())[-1]].stride
        self.extra_net = self._build_extra_nets(extra_in_channels, extra_in_stride)
        # ========header net for orient, confidence, dimension======== #
        feature_shapes = self.OutShapeSpec
        header_in_channels = feature_shapes[list(feature_shapes.keys())[-1]].channels
        header_in_stride = feature_shapes[list(feature_shapes.keys())[-1]].stride
        header_in_size = get_shapes_by_stride(configs.INTENSOR_SIZE, header_in_stride)
        self.orientation, self.confidence , self.dimension = self._build_subnet_header(header_in_channels,
                                                                                       header_in_size,
                                                                                       configs.DEEPBOX3D.OUT_CHANNELS)
        self.orient_loss = OrientationLoss
        self.conf_loss = nn.CrossEntropyLoss()
        self.dim_loss = nn.MSELoss()
        multibin = MultiBin(configs.DEEPBOX3D.BINS, configs.DEEPBOX3D.OVERLAP, device=self.device)
        self._binCoder = BinCoder(multibin, configs)

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
        stride = 1
        if self.configs.EXTRANET.USE_INV_RES:
            for _ in range(self.configs.EXTRANET.NUMLAYERS):
                extra = [InvertedResidual(in_channels if i == 0 else self.configs.DEEPBOX3D.OUT_CHANNELS,
                                          self.configs.DEEPBOX3D.OUT_CHANNELS, stride if i == 0 else 1,
                                          expand_ratio=6, use_batch_norm=True)
                         for i in range(self.configs.EXTRANET.NUMCONVS)]
                extra = nn.Sequential(*extra)
                extras.append(extra)
                in_channels = self.configs.DEEPBOX3D.OUT_CHANNELS
        else:
            for _ in range(self.configs.EXTRANET.NUMLAYERS):
                extra = [nn.Sequential(nn.Conv2d(in_channels if i == 0 else self.configs.DEEPBOX3D.OUT_CHANNELS,
                                                 self.configs.DEEPBOX3D.OUT_CHANNELS, 1, 1, padding=0),
                                       nn.BatchNorm2d(self.configs.DEEPBOX3D.OUT_CHANNELS),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.configs.DEEPBOX3D.OUT_CHANNELS, self.configs.DEEPBOX3D.OUT_CHANNELS,
                                                 3, stride if i == 0 else 1, padding=1),
                                       nn.BatchNorm2d(self.configs.DEEPBOX3D.OUT_CHANNELS),
                                       nn.ReLU(inplace=True))
                         for i in range(self.configs.EXTRANET.NUMCONVS)]
                extra = nn.Sequential(*extra)
                extras.append(extra)
                in_channels = self.configs.DEEPBOX3D.OUT_CHANNELS
        self._extra_feature_strides = self._extra_feature_strides + [in_stride * (stride ** (i + 1))
                                                                     for i in range(self.configs.EXTRANET.NUMLAYERS)]
        self._extra_feature_channels = self._extra_feature_channels + \
                                       [self.configs.DEEPBOX3D.OUT_CHANNELS] * self.configs.EXTRANET.NUMLAYERS
        return nn.Sequential(*extras)

    def _build_subnet_header(self, in_channels, in_size, out_channels):
        HxW = in_size[0]*in_size[1]
        orientation = nn.Sequential(
            nn.Linear(in_channels * HxW, out_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_channels, self.bins * 2)  # to get sin and cos
        )
        confidence = nn.Sequential(
            nn.Linear(in_channels * HxW, out_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_channels, self.bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )
        d_channels = out_channels * 2
        dimension = nn.Sequential(
            nn.Linear(in_channels * HxW, d_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(d_channels, d_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(d_channels, 3)
        )
        return orientation, confidence, dimension

    def forward(self, x, gt_orient=None, gt_conf=None,  gt_dim=None, is_training=False):
        x = self.base_net(x)
        x = self.extra_net(x)
        _, c, h, w = list(x.size())
        x = x.view(-1, c * h * w)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        if is_training:
            return self.loss(orientation, confidence, dimension,
                             gt_orient, gt_conf, gt_dim)
        return orientation, confidence, dimension

    def loss(self, pred_orient, pred_conf,  pred_dim,  gt_orient, gt_conf,  gt_dim):
        orient_loss = self.orient_loss(pred_orient, gt_orient, gt_conf)
        dim_loss = self.dim_loss(pred_dim, gt_dim)

        gt_conf = torch.max(gt_conf, dim=1)[1]
        conf_loss = self.conf_loss(pred_conf, gt_conf)

        loss_theta = conf_loss + self.w * orient_loss
        loss = self.alpha * dim_loss + loss_theta
        return loss

    @property
    def OutShapeSpec(self):
        extra_shapes = OrderedDict()

        for i, s in enumerate(self._extra_feature_strides):
            extra_shapes['extra_layer{}'.format(i)] = ShapeSpec(channels=self._extra_feature_channels[i],
                                                                stride=s)
        feature_shapes = OrderedDict(self.base_net.OutShapeSpec, **extra_shapes)
        return feature_shapes

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=False)
        self.extra_net.apply(_xavier_init_)
        self.orientation.apply(_xavier_init_)
        self.confidence.apply(_xavier_init_)
        self.dimension.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extra_net.apply(_xavier_init_)
        self.orientation.apply(_xavier_init_)
        self.confidence.apply(_xavier_init_)
        self.dimension.apply(_xavier_init_)

def _xavier_init_(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)