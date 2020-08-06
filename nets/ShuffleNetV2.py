import torch
import torch.hub as hub
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from nets.ShapeSpec import ShapeSpec

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ShuffleNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(ShuffleNetV2, self).__init__()
        self.model = hub.load('pytorch/vision:v0.5.1', 'shufflenet_v2_x1_0', pretrained=pretrained)
        self._out_features_channels = [116, 232, 464, 1024]
        self._out_features_strides = [2 ** (i+1) for i in range(2, 5)] + [2**5]


    def forward(self, x):
        self.features = OrderedDict()
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        self.features['stage_2'] = x
        x = self.model.stage3(x)
        self.features['stage_3'] = x
        x = self.model.stage4(x)
        self.features['stage_4'] = x
        x = self.model.conv5(x)
        self.features['conv_5'] = x
        return x

    @property
    def OutShapeSpec(self):
        specs = OrderedDict()
        for i in range(len(self._out_features_channels) - 1):
            specs['stage_{}'.format(i + 2)] = ShapeSpec(channels=self._out_features_channels[i],
                                                    stride=self._out_features_strides[i])
        specs['conv_5'] = ShapeSpec(channels=self._out_features_channels[-1],
                                                    stride=self._out_features_strides[-1])
        return specs

# model = ShuffleNetV2()
# x = Variable(torch.randn([1, 3, 160, 544]).type(torch.FloatTensor),
#                          requires_grad=False)
# model(x)
#
# print(model.features)
