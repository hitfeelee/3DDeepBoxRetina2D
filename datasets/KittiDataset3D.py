import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from collections import OrderedDict
from Archs_3D.MultiBin import MultiBin
from preprocess.data_preprocess_3d import TrainAugmentation
from Archs_3D.configs.mobilev2_deepbox3d_configs import CONFIGS
from datasets.distributed_sampler import *
from postprocess.visualization import cv_draw_bboxes_3d
from datasets.dataset_utils import AspectRatioGroupedDataset
import operator

# TODO: clean up where this is

class KittiDataset3D(data.Dataset):
    def __init__(self, path, transform=None,input_size=(288, 288)):

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        self.input_size = input_size
        self.transform = transform
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # hold average dimensions
        self.class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc',
                          'Background']
        self.label_map = OrderedDict({'{}'.format(c): i for i, c in enumerate(self.class_list)})

        self.object_list = self.get_objects(self.ids)
        # pre-fetch all labels
        self.labels = self.get_labels()

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # should return (Input, Label)
    def __getitem__(self, index):
        assert(self.object_list is not None)
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%id)

        label = self.labels[id][str(line_num)]
        gt_bbox = label['Box_2D']
        if self.transform is not None:
            img, gt_bbox, _ = self.transform(self.curr_img, bboxes=gt_bbox)
            label['Box_2D'] = gt_bbox
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            process = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            # crop image
            img = self.curr_img[gt_bbox[1]:gt_bbox[3] + 1, gt_bbox[0]:gt_bbox[2] + 1]
            img = cv2.resize(src=img, dsize=self.input_size, interpolation=cv2.INTER_CUBIC)
            # recolor, reformat
            img = process(img)
        return img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue
                    objects.append((id, line_num))
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_labels(self):
        labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                labels[id] = {}
                last_id = id

            labels[id][str(line_num)] = label

        return labels
    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        Box_2D = np.array([int(round(line[4])), int(round(line[5])), int(round(line[6])), int(round(line[7]))],
                          dtype=np.float32)

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.float32) # height, width, length
        # modify for the average
        # Dimension -= np.array(self.dim_mean[int(self.label_map[Class])])

        Location = [line[11], line[12], line[13]] # x, y, z
        # Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        label = {
                'Class': int(self.label_map[Class]),
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Location': Location,
                'Alpha': Alpha,
                'Ry':Ry
                }

        return label

    # def format_label(self, line):
    #     line = line[:-1].split(' ')
    #
    #     Class = line[0]
    #
    #     for i in range(1, len(line)):
    #         line[i] = float(line[i])
    #
    #     Alpha = line[3] # what we will be regressing
    #     Ry = line[14]
    #     Box_2D = [int(round(line[4])), int(round(line[5])), int(round(line[6])), int(round(line[7]))]
    #
    #     Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
    #     # modify for the average
    #     Dimension -= np.array(self.dim_mean[int(self.label_map[Class])])
    #
    #     Location = [line[11], line[12], line[13]] # x, y, z
    #     Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object
    #
    #     Orientation = np.zeros((self.multibin.bin_num, 2))
    #     Confidence = np.zeros(self.multibin.bin_num)
    #
    #     # alpha is [-pi..pi], shift it to be [0..2pi]
    #     angle = Alpha + np.pi
    #
    #     bin_idxs = self.multibin.get_bin(angle)
    #
    #     for bin_idx in bin_idxs:
    #         angle_diff = angle - self.multibin.get_bin_bench_angle(bin_idx)
    #         Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
    #         Confidence[bin_idx] = 1
    #
    #     label = {
    #             'Class': Class,
    #             'Box_2D': Box_2D,
    #             'Dimensions': Dimension,
    #             'Alpha': Alpha,
    #             'Orientation': Orientation,
    #             'Confidence': Confidence,
    #             'Ry':Ry
    #             }
    #
    #     return label

    def get_object(self, id):
        img = cv2.imread(self.top_img_path + '%s.png' % id)
        label = self.labels[id]
        return img, label

    def calc_dim_mean(self):
        dim_sum_map = {}
        class_sum_map = {}
        for id in self.ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue
                    Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)  # height, width, length
                    if obj_class not in class_sum_map:
                        class_sum_map[obj_class] = 0
                        dim_sum_map[obj_class] = np.zeros((3,), dtype=np.double)
                    else:
                        class_sum_map[obj_class] += 1
                        dim_sum_map[obj_class] += Dimension

        dim_mean_map = [dim_sum_map[c]/class_sum_map[c] for c in self.class_list[:-1]]
        print('dim mean: ', dim_mean_map)
        print('dim class:', class_sum_map)
"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch

if __name__ == '__main__':
    multibin = MultiBin(2, 0.1)
    dataset = KittiDataset3D('/home/fee/code/ai/datasets/kitti/training',multibin,
                               transform=TrainAugmentation((224, 224)), dim_mean=CONFIGS.DATASET.DIM_MEAN)
    num = len(dataset)
    # dataset.calc_dim_mean()
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
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        # image = image.astype(np.uint8)
        cv2.imshow("preprocessed image", image)
        cv2.waitKey(3000)