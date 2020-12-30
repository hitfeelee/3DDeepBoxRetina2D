import cv2
import numpy as np
import os
import operator
from torch.utils import data
from collections import OrderedDict
from preprocess.data_preprocess import TrainAugmentation
from postprocess.visualization import cv_draw_bbox_3d
from datasets.dataset_utils import AspectRatioGroupedDataset
from datasets.dataset_utils import Label
from datasets.distributed_sampler import *
from utils.ParamList import ParamList
from Archs_3D.configs.mobilev2_retina3d_configs import CONFIGS

class KITTI(data.Dataset):
    def __init__(self, cfg, transform=None, is_training=True):
        super(KITTI).__init__()
        self.root = cfg.DATASET.PATH
        self.image_dir = os.path.join(self.root, "image_2")
        self.label_dir = os.path.join(self.root, "label_2")
        self.calib_dir = os.path.join(self.root, "calib")

        self.split = cfg.DATASET.TRAIN_SPLIT if is_training else cfg.DATASET.TEST_SPLIT
        self.is_training = is_training
        self.transform = transform
        self.K = np.array(cfg.DATASET.INTRINSIC, dtype=np.float32)
        self.K = self.K.reshape((3, 4))

        if self.split == "train":
            imageset_txt = os.path.join(self.root, "ImageSets", "train.txt")
        elif self.split == "val":
            imageset_txt = os.path.join(self.root, "ImageSets", "val.txt")
        elif self.split == "trainval":
            imageset_txt = os.path.join(self.root, "ImageSets", "trainval.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(self.root, "ImageSets", "test.txt")
        else:
            raise ValueError("Invalid split!")

        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        self.classes = cfg.DATASET.DETECT_CLASSES

        self.label_map = OrderedDict({'{}'.format(c): i for i, c in enumerate(self.classes)})

    # should return (Input, Label)
    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.image_dir, self.image_files[index]))
        targets = ParamList(image_size=(img.shape[1], img.shape[0]), is_train=self.is_training)
        targets.add_field('K', np.copy(self.K))
        if not self.is_training:
            if self.transform is not None:
                img, targets = self.transform(img, targets=targets)
            return img, targets
        labels = self.parse_label(os.path.join(self.label_dir, self.label_files[index]))
        targets.add_field('class', labels[0])
        targets.add_field('bbox', labels[1])
        targets.add_field('dimension', labels[2])
        targets.add_field('alpha', labels[3])
        targets.add_field('Ry', labels[4])
        targets.add_field('mask', labels[5])


        if self.transform is not None:
            img, targets = self.transform(img, targets=targets)
        return img,  targets

    def __len__(self):
        return self.num_samples

    def parse_label(self, label_path):
        gt_classes = []
        gt_bboxes = []
        gt_dims = []
        gt_alphas = []
        gt_rays = []

        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class not in self.classes:
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Box_2D = [float(line[4]), float(line[5]), float(line[6]), float(line[7])]
                gt_classes.append(self.label_map[Class])
                gt_bboxes.append(Box_2D)
                gt_dims.append([float(line[8]), float(line[9]), float(line[10])]) # H, W, L
                gt_alphas.append(float(line[3])) #alpha
                gt_rays.append(float(line[-1])) #ry
            gt_classes = np.array(gt_classes, dtype=np.float32)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_dims = np.array(gt_dims, dtype=np.float32)
            gt_alphas = np.array(gt_alphas, dtype=np.float32)
            reg_masks = np.zeros((gt_classes.shape[0],), dtype=np.float32) + 0.
            gt_rays = np.array(gt_rays, dtype=np.float32)

        return gt_classes, gt_bboxes, gt_dims, gt_alphas, gt_rays, reg_masks

    def calc_mean_of_dataset(self):
        rgb_mean = np.zeros((3,), dtype=np.float64)
        num = self.num_samples
        for i, file in enumerate(self.image_files):
            img = cv2.imread(os.path.join(self.image_dir, file))
            rgb_mean += np.mean(img, axis=(0, 1))
            print("progress : {:.2%}".format(i/num))
        print('rgb mean: ', rgb_mean/num)



from postprocess.postprocessing import calc_regressed_bbox_3d
from postprocess.visualization import KITTI_COLOR_MAP
if __name__ == '__main__':

    dataset = KITTI(CONFIGS,transform=TrainAugmentation(CONFIGS.INTENSOR_SIZE, mean=CONFIGS.DATASET.MEAN), is_training=True)
    # dataset.calc_mean_of_dataset()
    sampler = TrainingSampler(len(dataset))
    params = {'sampler': sampler,
              'batch_sampler': None,
              'collate_fn': operator.itemgetter(0),  # don't batch, but yield individual elements
              'num_workers': 1}

    generator = data.DataLoader(dataset, **params)
    generator = AspectRatioGroupedDataset(generator, 1)

    for d in generator:
        # print(d)
        image = d[0][0]
        # image = image.permute(1,2,0).numpy().astype(np.uint8)
        img = image.astype(np.uint8)
        h, w, _ = img.shape
        targets = d[0][1]
        classes = targets.get_field('class').cpu().numpy().astype(np.long)
        bboxes = targets.get_field('bbox').cpu().numpy()
        alphas = targets.get_field('alpha').cpu().numpy()
        K = targets.get_field('K').cpu().numpy()
        dims = targets.get_field('dimension').cpu().numpy()
        centers = np.concatenate([(bboxes[:, None, 0] + bboxes[:, None, 2]) * .5,
                                 (bboxes[:, None, 1] + bboxes[:, None, 3]) * .5], axis=1)
        rays = np.arctan((centers[:, 0] - K[0, 2])/K[0, 0])
        num = alphas.shape[0]
        K[:, 3] = 0.
        for i in range(num):
            loc, _ = calc_regressed_bbox_3d(alphas[i], theta_ray=rays[i], dimension=dims[i], bboxes=bboxes[i], proj_matrix=K)
            proj_matrix = np.copy(K)
            proj_matrix[0, :] *= w
            proj_matrix[1, :] *= h
            cv_draw_bbox_3d(img, proj_matrix,
                        alphas[i] + rays[i], dims[i], loc, classes[i], 1., KITTI_COLOR_MAP[classes[i]])
        # img = cv_draw_bboxes_2d(img, bboxes, classes, dataset.classes)
        cv2.imshow("preprocessed image", img)
        cv2.waitKey(1000)