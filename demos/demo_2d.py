import __init__
from Archs_2D.DefaultPredictor import DefaultPredictor
from Archs_2D.build_model import build_model
import os
import cv2
from postprocess.visualization import draw_instance_to_image
import numpy as np
from datasets.KittiDatasets import KittiDatasets

from multiprocessing import Process, Pipe
import time
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="Retina Demo")
    parser.add_argument("--model-name", default="MOBI-V2-RETINA-FPN", help="specific model name")
    return parser

class Demo2D(Process):
    def __init__(self, model_name, type='video', pipe=None):
        super(Demo2D, self).__init__()
        self.model_name = model_name
        self.type = type
        self.pipe = pipe
        model, backbone, cfg = build_model(self.model_name)
        self.predictor = DefaultPredictor(model, cfg)
        self.cfg = cfg

    def run(self):
        #=====test images kitti dataset========#
        kitti_dir = os.path.join(self.cfg.DATASET.PATH, 'testing')
        image_dir = os.path.join(self.cfg.DATASET.PATH, 'testing/image_2')
        kitti = KittiDatasets(kitti_dir)
        if 'video' == self.type:
            self.detect_objects_from_video("/home/fee/Videos/road.mp4", self.predictor, kitti.class_list)
        else:
            self.detect_objects_from_images(image_dir, self.predictor, kitti.class_list)

    def detect_objects_from_images(self, path, predictor, label_map=None):
        image_list = os.listdir(path)
        for img in image_list:
            img = cv2.imread(os.path.join(path, img))
            src = np.copy(img)
            start_time = time.time()
            outputs = predictor(img)
            end_time = time.time()
            print('detecting time %s per image' % (end_time - start_time))
            draw_instance_to_image(src, outputs[0]['instances'], label_map=label_map)
            cv2.imshow('DETECTOR RESULT', src)
            k = cv2.waitKey(1000)
            if (k & 0xff == ord('q')):
                break

    def detect_objects_from_video(self, path, predictor, label_map=None):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(cap.isOpened())
        rx = 8.
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                shape = np.shape(frame)
                frame = frame[int(shape[0] / 3.): shape[0], int(shape[1] / rx): int((rx - 1.) * shape[1] / rx)]
                src = np.copy(frame)
                shape = np.shape(frame)

                start_time = time.time()
                outputs = predictor(frame)
                end_time = time.time()
                # self.pipe.send({
                #     'image': np.copy(src),
                #     'instances': outputs[0]['instances']
                # })
                # self.pipe.send({
                #     'image': 1
                # })
                print('detecting time %s per image' % (end_time - start_time))
                draw_instance_to_image(src, outputs[0]['instances'], label_map=label_map)
                k = cv2.waitKey(10)
                if (k & 0xff == ord('q')):
                    break
                cv2.imshow('DETECTOR RESULT', src)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pipe_node1, pipe_node2 = Pipe()
    args = arg_parser().parse_args()
    demo = Demo2D(model_name=args.model_name, type='image')
    demo.run()
    # while True:
    #     print(pipe_node2.recv())
    #     time.sleep(1)
    #     # cv2.waitKey(100)
