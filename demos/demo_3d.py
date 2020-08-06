import __init__
import torch
from Archs_3D.DefaultPredictor import DefaultPredictor
from Archs_3D.build_model import build_model
from postprocess.visualization import *
import time
from datasets.KittiDataset3D import KittiDataset3D
from multiprocessing import Process
import argparse
import os



def arg_parser():
    parser = argparse.ArgumentParser(description="Deepbox3D Demo")
    parser.add_argument("--model-name", default="MOBI-V2-DEEPBOX3D", help="specific model name")
    return parser

class Demo3D(Process):
    def __init__(self, model_name="MOBI-V2-DEEPBOX3D", type='image', pipe=None):
        super(Demo3D, self).__init__()
        self.model_name = model_name
        self.type = type
        self.pipe = pipe
        model, _, cfg = build_model(model_name)
        self.predictor = DefaultPredictor(model, cfg)
        self.cfg = cfg

    def run(self):
        if self.type == 'pipe':
            self.detect_objects_from_pipe()
        else:
            self.detect_objects_from_images()

    def detect_objects_from_images(self):
        path = os.path.join(self.cfg.DATASET.PATH, 'training')
        kitti = KittiDataset3D(path)
        for id in kitti.ids:
            img, label = kitti.get_object(id)
            classes = [torch.tensor(d['Class'], dtype=torch.long) for d in label.values()]
            bboxes = [torch.from_numpy(d['Box_2D']) for d in label.values()]
            classes = torch.stack(classes, dim=0)
            bboxes = torch.stack(bboxes, dim=0)
            locations, orients, dimensions = self.predictor(img, bboxes, classes)
            for loc, orient, dim in zip(locations, orients, dimensions):
                plot_3d_box(img, self.predictor.proj_matrix.cpu().numpy(),
                            orient, dim, loc)
            cv2.imshow('3D DETECTOR RESULT', img)
            k = cv2.waitKey(2000)
            if (k & 0xff == ord('q')):
                break

    def detect_objects_from_pipe(self):
        while True:
            result = self.pipe.recv()
            image = result['image']
            instance = result['instance']
            print(result)
            time.sleep(1)
            # cv2.waitKey(10)
            pass


if __name__ == '__main__':
    args = arg_parser().parse_args()
    demo = Demo3D(model_name=args.model_name, type='image')
    demo.run()
