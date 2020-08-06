import torch
from Archs_3D.configs import config
from nets.Backbone import build_backbone

from Archs_3D.DeepBox3D import DeepBox3DArch
import numpy as np

def build_model(name):
    backbone = None
    cfg = None
    if name == 'MOBI-V2-DEEPBOX3D':
        backbone = build_backbone('MOBI-V2')
        cfg =  config.get_model_config('MOBI-V2-DEEPBOX3D')
    elif name == 'SHUFFLE-DEEPBOX3D':
        backbone = build_backbone('SHUFFLE')
        cfg = config.get_model_config('SHUFFLE-DEEPBOX3D')
    else:
        assert backbone is not None
    model = DeepBox3DArch(backbone, cfg)
    return model, backbone, cfg

if __name__ == '__main__':
    model, backbone, cfg = build_model('MOBI-V2-DEEPBOX3D')
    input = torch.tensor(np.ones((1, 3, cfg.INTENSOR_SHAPE[0], cfg.INTENSOR_SHAPE[1]), dtype=np.float), dtype=torch.float32)
    logits, bboxes = model(input)
    print(logits)
    print(bboxes)