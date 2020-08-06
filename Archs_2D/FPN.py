import torch.nn as nn
from nets.Backbone import BackBone
from nets.ShapeSpec import ShapeSpec
from nets.MobilenetV2 import InvertedResidual
from collections import OrderedDict
from typing import Dict, List
from nets import Layer

def assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class FPN(BackBone):
    def __init__(self, feature_shapes:Dict[str, ShapeSpec], out_channels:int, configs=None):
        super(FPN, self).__init__(configs)
        assert(configs.FPN.FUSE_TYPE in {'avg', 'sum'})
        self._fuse_type = configs.FPN.FUSE_TYPE
        self.out_channels = out_channels
        self.in_features = configs.FPN.IN_FEATURES
        in_shapes = [feature_shapes[feature] for feature in configs.FPN.IN_FEATURES]
        in_strides = [s.stride for s in in_shapes]
        in_channels = [s.channels for s in in_shapes]
        assert_strides_are_log2_contiguous(in_strides)

        lateral_convs = []
        output_convs = []

        for i, in_chs in enumerate(in_channels):
            lateral_conv = nn.Conv2d(in_chs, out_channels, kernel_size=1, bias=True)
            if self.configs.EXTRANET.USE_INV_RES:
                OutputLayer = InvertedResidual
                params = {'stride': 1, 'expand_ratio': 6, 'use_batch_norm': True}
            else:
                OutputLayer = Layer.Conv2d
                params = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True,
                          'norm': nn.BatchNorm2d(out_channels)}
            output_conv = OutputLayer(out_channels, out_channels, **params)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = nn.Sequential(*lateral_convs[::-1])
        self.output_convs = nn.Sequential(*output_convs[::-1])
        self.upsample = nn.Upsample(scale_factor=2, mode=configs.FPN.UPSAMPLE_TYPE)
        self.in_strides = in_strides
        pass

    def forward(self, x:OrderedDict)->List:
        x = [x[f] for f in self.in_features[::-1]]
        result = []
        prev_features = self.lateral_convs[0](x[0])
        result.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv in zip(x[1:], self.lateral_convs[1:], self.output_convs[1:]):
            up_features = self.upsample(prev_features)
            lateral_features = lateral_conv(features)
            prev_features = up_features + lateral_features
            if self._fuse_type == 'avg':
                prev_features /= 2
            result.insert(0, output_conv(prev_features))
        self.features = OrderedDict({'fpn_layer{}'.format(i): result[i] for i in range(len(result))})
        return result

    @property
    def OutShapeSpec(self):
        specs = OrderedDict({'fpn_layer{}'.format(i): ShapeSpec(channels=self.out_channels,
                                                    stride=s)
                 for i, s in enumerate(self.in_strides)})
        return specs
