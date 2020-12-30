from fvcore.common.config import CfgNode
from Archs_3D import Register

CONFIGS = CfgNode()
CONFIGS.INTENSOR_SIZE = (288, 288)
CONFIGS.BATCH_SIZE = 8
CONFIGS.DEVICE = 'cpu'

CONFIGS.TRAINING = CfgNode()
CONFIGS.TRAINING.LOGDIR = './logdirs/mobiv2_retina3d'
CONFIGS.TRAINING.CHECKPOINT_MODE = 'RESUME'  # ['PRETRAINED', 'RESUME', 'START']
CONFIGS.TRAINING.CHECKPOINT_FILE = './weights/mobilenetv2_retina2d_fpn_model.pth'

CONFIGS.DATASET = CfgNode()
CONFIGS.DATASET.PATH = './datasets/data/kitti/training'
CONFIGS.DATASET.MEAN = [95.87739305, 98.76049672, 93.83309082]
CONFIGS.DATASET.TRAIN_SPLIT = 'train'
CONFIGS.DATASET.TEST_SPLIT = 'test'
CONFIGS.DATASET.DETECT_CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'Background')
CONFIGS.DATASET.INTRINSIC = (7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03)
CONFIGS.DATASET.DIM_MEAN = [[1.52607842, 1.62858147, 3.88396124],
                            [2.20649159, 1.90197734, 5.07812564],
                            [3.25207685,  2.58505032, 10.10703568],
                            [1.76067766, 0.6602296 , 0.84220464],
                            [1.27538462, 0.59506787, 0.80180995],
                            [1.73712792, 0.59677122, 1.76338868],
                            [3.52905882,  2.54368627, 16.09707843],
                            [1.9074177, 1.51386831, 3.57683128]]



CONFIGS.DATALOADER = CfgNode()
CONFIGS.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'
# Solver
# ---------------------------------------------------------------------------- #
CONFIGS.SOLVER = CfgNode()

# See detectron2/solver/build.py for LR scheduler options
CONFIGS.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

CONFIGS.SOLVER.MAX_ITER = 1000000

CONFIGS.SOLVER.BASE_LR = 0.0025

CONFIGS.SOLVER.MOMENTUM = 0.9

CONFIGS.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
CONFIGS.SOLVER.WEIGHT_DECAY_NORM = 0.0

CONFIGS.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
CONFIGS.SOLVER.STEPS = (500000, 800000)

CONFIGS.SOLVER.WARMUP_FACTOR = 1.0 / 1000
CONFIGS.SOLVER.WARMUP_ITERS = 1000
CONFIGS.SOLVER.WARMUP_METHOD = "linear"
CONFIGS.SOLVER.EXCLUDE_SCOPE = ('base_net', 'extras', 'fpn', 'classification_headers', 'bbox_regress_headers')


# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
CONFIGS.SOLVER.BIAS_LR_FACTOR = 1.0
CONFIGS.SOLVER.WEIGHT_DECAY_BIAS = CONFIGS.SOLVER.WEIGHT_DECAY
CONFIGS.SOLVER.LOAD_SOLVER = True
CONFIGS.MOBILENETV2 = CfgNode()

CONFIGS.EXTRANET = CfgNode()
CONFIGS.EXTRANET.NUMLAYERS = 2
CONFIGS.EXTRANET.NUMCONVS = 1
CONFIGS.EXTRANET.USE_INV_RES = False
CONFIGS.EXTRANET.EXPAND_RATIO = 1

CONFIGS.RETINANET = CfgNode()
CONFIGS.RETINANET.NUM_CLASSES = 8
CONFIGS.RETINANET.OUT_CHANNELS = 256
CONFIGS.RETINANET.PRIOR_PROB = 0.01
CONFIGS.RETINANET.HEADER_NUMCONVS = 2
CONFIGS.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
CONFIGS.RETINANET.IOU_LABELS = [0, -1, 1]
CONFIGS.RETINANET.OD_FEATURES = ['fpn_layer0', 'fpn_layer1', 'fpn_layer2', 'extra_layer1', 'extra_layer2']
CONFIGS.RETINANET.FOCAL_LOSS_GAMMA = 2.0
CONFIGS.RETINANET.FOCAL_LOSS_ALPHA = 0.25
CONFIGS.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1

CONFIGS.RPN = CfgNode()
CONFIGS.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

CONFIGS.ANCHOR_GENERATOR = CfgNode()
CONFIGS.ANCHOR_GENERATOR.SIZES = [[x, x * 2**(1.0/3), x * 2**(2.0/3)] for x in [10, 20, 40, 80, 160]]
CONFIGS.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
CONFIGS.ANCHOR_GENERATOR.OFFSET = 0.0

CONFIGS.FPN = CfgNode()
CONFIGS.FPN.IN_FEATURES = ['mobilev2_layer6', 'mobilev2_layer13', 'mobilev2_layer18']
CONFIGS.FPN.FUSE_TYPE = 'sum' #['avg', 'sum']
CONFIGS.FPN.UPSAMPLE_TYPE = 'nearest' #['nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear']
CONFIGS.FPN.USE_INV_RES = False

CONFIGS.DETECTOR = CfgNode()
CONFIGS.DETECTOR.CHECKPOINT = './weights/mobilenetv2_retina3d_fpn_model.pth'
CONFIGS.DETECTOR.SCORE_THRESH_TEST = 0.3
CONFIGS.DETECTOR.TOPK_CANDIDATES_TEST = 1000
CONFIGS.DETECTOR.NMS_THRESH_TEST = 0.5
CONFIGS.DETECTOR.DETECTIONS_PER_IMAGE = 100

CONFIGS.INPUT = CfgNode()
CONFIGS.INPUT.FORMAT = 'BGR'

CONFIGS.MULTIBIN = CfgNode()
CONFIGS.MULTIBIN.BINS = 2
CONFIGS.MULTIBIN.OVERLAP = 0.1
CONFIGS.MULTIBIN.ALPHA = 0.6
CONFIGS.MULTIBIN.W = 0.4

@Register.Config.register('MOBI-V2-RETINA3D-FPN')
def register():
    return CONFIGS


