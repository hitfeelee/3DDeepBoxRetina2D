
import torch
from preprocess.data_preprocess_3d import PredictionTransform
import numpy as np
class DetectedObject(object):
    def __init__(self, img, classes, box_2d, proj_matrix, input_size=(224, 224)):

        self.transform = PredictionTransform(input_size)
        self.proj_matrix = proj_matrix
        self.theta_rays = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.croped_imgs, self.bboxes= self.format_img(img, box_2d)
        self.classes = classes


    def calc_theta_ray(self, img, box_2d, proj_matrix):
        bboxes = box_2d
        if not torch.is_tensor(box_2d):
            bboxes = torch.from_numpy(box_2d)
        width = img.shape[1]
        fovx = 2 * torch.atan(width / (2 * proj_matrix[0][0]))
        center = (bboxes[:, 0] + bboxes[:, 2]) / 2
        dx = center - (width / 2)
        mult = torch.ones(dx.size(), dtype=torch.float32)
        mult = torch.where(dx >= 0, mult, -1.*mult)
        dx = torch.abs(dx)
        angle = torch.atan( (2*dx*torch.tan(fovx/2)) / width )
        angle = angle * mult
        return angle

    def format_img(self, img, box_2d):
        if torch.is_tensor(box_2d):
            bboxes = box_2d.cpu().numpy()
        else:
            bboxes = box_2d
        # crop image
        croped_imgs, bboxes, _ = self.transform(img, bboxes=bboxes)
        croped_imgs = torch.stack(croped_imgs, dim=0)
        return croped_imgs, bboxes