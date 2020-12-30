import cv2
import numpy as np
import os
import operator
from torch.utils import data
from collections import OrderedDict
from preprocess.data_preprocess import TrainAugmentation
from postprocess.visualization import cv_draw_bboxes_2d
from datasets.dataset_utils import AspectRatioGroupedDataset
from datasets.dataset_utils import Label
from datasets.distributed_sampler import *

class KittiDatasets(data.Dataset):
    def __init__(self, path, transform = None):
        super(KittiDatasets).__init__()
        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)
        self.transform = transform
        # hold average dimensions
        self.class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc', 'Background']
        self.label_map = OrderedDict({'{}'.format(c): i for i, c in enumerate(self.class_list)})

    # should return (Input, Label)
    def __getitem__(self, index):
        id = self.ids[index]
        img = cv2.imread(self.top_img_path + '%s.png'%id)
        gt_classes, gt_bboxes = self.parse_label(self.top_label_path + '%s.txt'%id)
        if self.transform is not None:
            img, gt_bboxes, gt_classes = self.transform(img, bboxes=gt_bboxes, labels=gt_classes)
        return {'image':img, 'labels':Label(gt_bboxes, gt_classes)}

    def __len__(self):
        return len(self.ids)

    def parse_label(self, label_path):
        gt_classes = []
        gt_bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Box_2D = [float(line[4]), float(line[5]), float(line[6]), float(line[7])]
                gt_classes.append(self.label_map[Class])
                gt_bboxes.append(Box_2D)
            gt_classes = np.array(gt_classes, dtype=np.float32)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        return gt_classes, gt_bboxes

    def calc_mean_of_dataset(self):
        rgb_mean = np.zeros((3,), dtype=np.float64)
        num = len(self.ids)
        for i, id in enumerate(self.ids):
            img = cv2.imread(self.top_img_path + '%s.png' % id)
            rgb_mean += np.mean(img, axis=(0, 1))
            print("progress : {:.2%}".format(i/num))
        print('rgb mean: ', rgb_mean/num)

if __name__ == '__main__':
    dataset = KittiDatasets('/home/fee/code/ai/datasets/kitti/training',
                            transform=TrainAugmentation(224, [95.87739305, 98.76049672, 93.83309082]))
    # dataset.calc_mean_of_dataset()
    # dataset = KittiDatasets('/home/fee/code/ai/datasets/kitti/training',
    #                         transform=None)
    sampler = TrainingSampler(len(dataset))
    params = {'sampler': sampler,
              'batch_sampler': None,
              'collate_fn': operator.itemgetter(0),  # don't batch, but yield individual elements
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)
    generator = AspectRatioGroupedDataset(generator, 1)
    for d in generator:
        # print(d)
        image = d[0]['image']
        image = image.permute(1,2,0).numpy().astype(np.uint8)
        bboxes = d[0]['labels'].gt_bboxes.tensor.numpy()
        classes = d[0]['labels'].gt_classes
        # cv_draw_bboxes_2d(image, bboxes, classes, dataset.class_list)
        cv2.imshow("preprocessed image", image)
        cv2.waitKey(3000)