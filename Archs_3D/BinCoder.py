import torch
import numpy as np


class BinCoder(object):
    def __init__(self, multibin, cfg):
        self.multibin = multibin
        self.cfg = cfg
        self.dim_mean = torch.tensor(self.cfg.DATASET.DIM_MEAN, dtype=torch.float32)

    def encode(self, gt_alpha, gt_dim, gt_classes):
        batch_size = list(gt_classes.size())[0]
        self.dim_mean = self.dim_mean.to(gt_dim.device)
        dim_means = self.dim_mean[gt_classes]
        Dimension = gt_dim - dim_means

        Orientation = torch.zeros((batch_size, self.multibin.bin_num, 2), dtype=torch.float32, device=gt_dim.device)
        Confidence = torch.zeros((batch_size, self.multibin.bin_num,), dtype=torch.long, device=gt_dim.device)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = gt_alpha + np.pi

        bin_idxs = self.multibin.get_bins(angle)
        bin_ben_angles = self.multibin.get_bins_bench_angle(bin_idxs[1])
        angle_diff = angle[bin_idxs[0]] - bin_ben_angles
        Confidence[bin_idxs] = 1
        Orientation[bin_idxs[0],bin_idxs[1], 0] = torch.cos(angle_diff).to(torch.float32)
        Orientation[bin_idxs[0],bin_idxs[1], 1] = torch.sin(angle_diff).to(torch.float32)

        return Orientation, Confidence, Dimension


    def decode(self, pred_orient, pred_conf, pred_dim, pred_class):
        batch_size = list(pred_class.size())[0]
        dim_means = self.dim_mean[pred_class]
        Dimension = pred_dim + dim_means

        argmax = torch.argmax(pred_conf, dim=1)
        indexs = torch.tensor(range(batch_size), dtype=torch.long, device=pred_orient.device)
        orient = pred_orient[indexs, argmax]
        Alpha = torch.atan2(orient[:, 1], orient[:, 0])
        Alpha += self.multibin.get_bin_bench_angle(argmax)
        Alpha -= np.pi
        return Alpha, Dimension



