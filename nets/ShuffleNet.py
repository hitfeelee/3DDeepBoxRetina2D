"""
implement a shuffleNet by pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

dtype = torch.FloatTensor
from collections import OrderedDict
from .ShapeSpec import ShapeSpec


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out


class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out


class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""

    def __init__(self, groups=3, in_channels=3):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage1_seq = [ShuffleNetUnitB(24, 60, groups=groups)] + \
                     [ShuffleNetUnitA(60, 60, groups=groups) for _ in range(5)]
        self.stage1 = nn.Sequential(*stage1_seq)
        stage2_seq = [ShuffleNetUnitB(60, 240, groups=groups)] + \
                     [ShuffleNetUnitA(240, 240, groups=groups) for _ in range(5)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=groups)] + \
                     [ShuffleNetUnitA(480, 480, groups=groups) for _ in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=groups)] + \
                     [ShuffleNetUnitA(960, 960, groups=groups) for _ in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)

        self._out_features_channels = [24, 60, 240, 480, 960]
        self._out_features_strides = [2 ** i for i in range(1, 6)]

    def forward(self, x):
        self.features = OrderedDict()
        net = self.conv1(x)

        # net = F.max_pool2d(net, 3, stride=2, padding=1)
        net = self.stage1(net)
        self.features['stage_1'] = net
        net = self.stage2(net)
        self.features['stage_2'] = net
        net = self.stage3(net)
        self.features['stage_3'] = net
        net = self.stage4(net)
        self.features['stage_4'] = net
        return net

    @property
    def OutShapeSpec(self):
        specs = OrderedDict()
        for i, layer in enumerate(self._out_features_channels):
            specs['stage_{}'.format(i)] = ShapeSpec(channels=self._out_features_channels[i],
                                                    stride=self._out_features_strides[i])
        return specs


if __name__ == "__main__":
    shuffleNet = ShuffleNet()
    shuffleNet.eval()
    for _ in range(10):
        with torch.no_grad():
            x = Variable(torch.randn([1, 3, 224, 224]).type(dtype),
                         requires_grad=False)
            time_st = time.time()
            out = shuffleNet(x)
            det_t = time.time() - time_st
            print('time: ', det_t)
            print(shuffleNet.OutShapeSpec)
