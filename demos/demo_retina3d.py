import __init__
import torch
from Archs_3D.Retina3DPredictor import Retina3DPredictor
from Archs_3D.build_retina3d_model import build_model
from preprocess.data_preprocess import PredictionTransform
from postprocess.visualization import *
import time
from datasets.kitti import KITTI
from multiprocessing import Process
import argparse
from preprocess.transforms import ToAbsoluteCoords
import os
import logging
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

def arg_parser():
    parser = argparse.ArgumentParser(description="Retina3D Demo")
    parser.add_argument("--model-name", default="MOBI-V2-RETINA3D-FPN", help="specific model name")
    return parser

class DemoRetina3D(Process):
    def __init__(self, model_name="MOBI-V2-RETINA3D-FPN", type='image'):
        super(DemoRetina3D, self).__init__()
        self.model_name = model_name
        self.type = type
        model, _, cfg = build_model(model_name)
        self.predictor = Retina3DPredictor(model, cfg)
        self.cfg = cfg
        self.to_abs_coord = ToAbsoluteCoords()

    def run(self):
        self.detect_objects_from_images()

    def detect_objects_from_images(self):
        kitti = KITTI(self.cfg, is_training=True)
        logger = logging.getLogger('demo')
        for img, target in kitti:
            src2d = np.copy(img)
            src3d = np.copy(img)
            start_time = time.time()
            result = self.predictor(img, target)
            end_time = time.time()
            logger.info('the time of mode inference: {:.3f}'.format(end_time - start_time))
            classes = result.get_field('class').cpu().numpy().astype(np.long)
            num = classes.shape[0]
            if num > 0:
                src, result = self.to_abs_coord(src3d, targets=result)
                cv_draw_bboxes_3d(src3d, result, kitti.classes)

                cv_draw_bboxes_2d(src2d, result, kitti.classes)
            numpy_vertical = np.concatenate((src2d, src3d), axis=0)
            cv2.imshow('3D DETECTOR RESULT', numpy_vertical)
            k = cv2.waitKey(2000)
            if (k & 0xff == ord('q')):
                break


if __name__ == '__main__':
    args = arg_parser().parse_args()
    demo = DemoRetina3D(model_name=args.model_name, type='image')
    demo.run()
