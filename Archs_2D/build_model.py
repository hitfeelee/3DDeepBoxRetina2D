import torch
from Archs_2D.configs import config
from nets.Backbone import build_backbone

from Archs_2D.RetinaNet import RetinaNet
import numpy as np

test_config = 'MOBIL-V2-RETINA-FPN'
test_backbone = 'MOBI-V2'

def build_model(name):
    backbone = None
    cfg = None
    if name == 'MOBI-V2-RETINA-FPN':
        backbone = build_backbone('MOBI-V2')
        cfg = config.get_model_config('MOBI-V2-RETINA-FPN')
    if name == 'MOBI-V3-RETINA-FPN':
        backbone = build_backbone('MOBI-V3')
        cfg = config.get_model_config('MOBI-V3-RETINA-FPN')
    elif name == 'SHUFFLE-RETINA-FPN':
        backbone = build_backbone('SHUFFLE')
        cfg = config.get_model_config('SHUFFLE-RETINA-FPN')
    else:
        assert backbone is not None
    model = RetinaNet(backbone, cfg)
    return model, backbone, cfg

if __name__ == '__main__':
    model, backbone, cfg = build_model('MOBI-V2-RETINA-FPN')
    input = torch.tensor(np.ones((1, 3,  cfg.INTENSOR_SHAPE[0], cfg.INTENSOR_SHAPE[1]), dtype=np.float), dtype=torch.float32)
    logits, bboxes = model(input)
    print(logits)
    print(bboxes)


