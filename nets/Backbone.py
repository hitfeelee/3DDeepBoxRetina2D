import torch.nn as nn
import torch
from collections import OrderedDict
from nets import Register
from .MobilenetV2 import MobileNetV2
from .MobilenetV3 import MobileNetV3
from .ShuffleNet import ShuffleNet
from .ShuffleNetV2 import ShuffleNetV2
class BackBone(nn.Module):
    def __init__(self, configs=None):
        super(BackBone, self).__init__()
        self._features = OrderedDict()
        self._configs = configs

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        if isinstance(features, OrderedDict):
            self._features = features

    @features.deleter
    def features(self):
        del self._features

    @property
    def configs(self):
        return self._configs


@Register.Backbone.register('MOBI-V2')
def build_mobi_v2_backbone():
    return MobileNetV2()

@Register.Backbone.register('MOBI-V3')
def build_mobi_v3_backbone():
    return MobileNetV3()

@Register.Backbone.register('SHUFFLE')
def build_shuffle_backbone():
    return ShuffleNetV2()



def build_backbone(name):
    return Register.Backbone[name]()