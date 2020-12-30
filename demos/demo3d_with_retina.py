import __init__
from Archs_2D.DefaultPredictor import DefaultPredictor as DefaultPredictor2D
from Archs_2D.build_model import build_model as build_model_2d

from Archs_3D.DefaultPredictor import DefaultPredictor as DefaultPredictor3D
from Archs_3D.build_model import build_model as build_model_3d

import os
from postprocess.visualization import *
import numpy as np
from datasets.KittiDatasets import KittiDatasets
from datasets.KittiDataset3D import KittiDataset3D
import time
import argparse
from preprocess.transforms import ToAbsoluteCoords

def arg_parser():
    parser = argparse.ArgumentParser(description="Retina-Deepbox3D Demo")
    parser.add_argument("--model-name-2D", default="MOBI-V2-RETINA-FPN", help="specific 2D model name")
    parser.add_argument("--model-name-3D", default="MOBI-V2-DEEPBOX3D", help="specific 3D model name")
    parser.add_argument("--type", default="image", help="specific what is detected")
    return parser

class Demo3D_with_Retina(object):
    def __init__(self, model_name_2d, model_name_3d, type='video'):
        super(Demo3D_with_Retina, self).__init__()
        self.type = type
        model2D, _, retina_cfg = build_model_2d(model_name_2d)
        self.predictor2D = DefaultPredictor2D(model2D, retina_cfg)

        model3D, _, deepbox3D_cfg = build_model_3d(model_name_3d)
        self.predictor3D = DefaultPredictor3D(model3D, deepbox3D_cfg)

        self.kitti2D = KittiDatasets(os.path.join(retina_cfg.DATASET.PATH, 'training'))
        self.kitti3D = KittiDataset3D(os.path.join(retina_cfg.DATASET.PATH, 'training'))
        self.retina_cfg = retina_cfg
        self.to_abs_coord = ToAbsoluteCoords()

    def run(self):
        #=====test images kitti dataset========#

        image_dir = os.path.join(self.retina_cfg.DATASET.PATH, 'training/image_2')
        if 'video' == self.type:
            self.detect_objects_from_video("/home/fee/Videos/road.mp4", label_map=self.kitti2D.class_list)
        else:
            self.detect_objects_from_images(image_dir, label_map=self.kitti2D.class_list)

    def detect_objects_from_images(self, path, label_map=None):
        image_list = os.listdir(path)
        for img in image_list:
            img = cv2.imread(os.path.join(path, img))
            src2d = np.copy(img)
            src3d = np.copy(img)
            start_time = time.time()
            outputs = self.predictor2D(img)
            classes = outputs.get_field('class')
            num = classes.size(0)
            if num > 0:
                src2d, outputs = self.to_abs_coord(src2d, targets=outputs)
                bboxes = outputs.get_field('bbox')
                locations, orients, dimensions = self.predictor3D(src3d, bboxes, classes)
                end_time = time.time()

                print('detecting time %s per image' % (end_time - start_time))
                cv_draw_bboxes_2d(src2d, outputs, label_map=label_map)
                clses = classes.cpu().numpy()
                for cls, loc, orient, dim in zip(clses, locations, orients, dimensions):
                    cv_draw_bbox_3d(src3d, self.predictor3D.proj_matrix.cpu().numpy(),
                                orient, dim, loc, cls, 1., KITTI_COLOR_MAP[cls])
            numpy_vertical = np.concatenate((src2d, src3d), axis=0)
            k = cv2.waitKey(3000)
            if (k & 0xff == ord('q')):
                break
            cv2.imshow('DETECTOR RESULT', numpy_vertical)
            # numpy_vertical = cv2.resize(numpy_vertical, (numpy_vertical.shape[1]//4, numpy_vertical.shape[0]//4))
            # cv2.imwrite('./demos/detect_result.jpeg', numpy_vertical)

    def detect_objects_from_video(self, path, label_map=None):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        print(cap.isOpened())
        rx = 8.
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                shape = np.shape(frame)
                frame = frame[int(shape[0] / 3.): shape[0], int(shape[1] / rx): int((rx - 1.) * shape[1] / rx)]
                src2d = np.copy(frame)
                src3d = np.copy(frame)
                start_time = time.time()
                outputs = self.predictor2D(frame)
                classes = outputs.get_field('class')
                num = classes.size(0)
                if num > 0:
                    src2d, outputs = self.to_abs_coord(src2d, targets=outputs)
                    bboxes = outputs.get_field('bbox')
                    locations, orients, dimensions = self.predictor3D(src3d, bboxes, classes)
                    end_time = time.time()

                    print('detecting time %s per image' % (end_time - start_time))
                    cv_draw_bboxes_2d(src2d, outputs, label_map=label_map)
                    clses = classes.cpu().numpy()
                    for cls, loc, orient, dim in zip(clses, locations, orients, dimensions):
                        cv_draw_bbox_3d(src3d, self.predictor3D.proj_matrix.cpu().numpy(),
                                        orient, dim, loc, cls, 1., KITTI_COLOR_MAP[cls])
                numpy_vertical = np.concatenate((src2d, src3d), axis=0)
                k = cv2.waitKey(1000)
                if (k & 0xff == ord('q')):
                    break
                cv2.imshow('DETECTOR RESULT', numpy_vertical)
                # cv2.imwrite('./demos/detect_result.png', numpy_vertical)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = arg_parser().parse_args()
    demo = Demo3D_with_Retina(args.model_name_2D, args.model_name_3D, type=args.type)
    demo.run()