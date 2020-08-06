
import random
import sys
from datetime import datetime
import torch
import numpy as np
import os
import logging
import torch.utils.data as data
import json

def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)

class Label(object):
    def __init__(self, gt_bboxes, gt_classes):
        self.gt_classes = gt_classes
        self.gt_bboxes = gt_bboxes

    def __len__(self):
        if isinstance(self.gt_classes, list):
            return len(self.gt_classes)
        elif isinstance(self.gt_classes, torch.Tensor):
            return list(self.gt_classes.size())[0]
        elif type(self.gt_classes) is np.ndarray:
            return self.gt_classes.shape[0]
        else:
            return 0

# class AspectRatioGroupedDataset(object):
#     """
#     Batch data that have similar aspect ratio together.
#     In this implementation, images whose aspect ratio < (or >) 1 will
#     be batched together.
#
#     It assumes the underlying dataset produces dicts with "width" and "height" keys.
#     It will then produce a list of original dicts with length = batch_size,
#     all with similar aspect ratios.
#     """
#
#     def __init__(self, dataset):
#         """
#         Args:
#             dataset: an iterable. Each element must be a dict with keys
#                 "width" and "height", which will be used to batch data.
#             batch_size (int):
#         """
#         self.dataset = dataset
#         self.batch_size = dataset.batch_size
#         self._buckets = [[] for _ in range(2)]
#         # Hard-coded two aspect ratio groups: w > h and w < h.
#         # Can add support for more aspect ratio groups, but doesn't seem useful
#
#     def __iter__(self):
#         for d in self.dataset:
#             _, h, w = list(d["image"].size())
#             bucket_id = 0 if w > h else 1
#             bucket = self._buckets[bucket_id]
#             bucket.append(d)
#             if len(bucket) == self.batch_size:
#                 yield bucket[:]
#                 del bucket[:]

class AspectRatioGroupedDataset(object):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        bucket = []
        for d in self.dataset:
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                bucket = []

"""
Enables writing json with numpy arrays to file
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)

"""
Class will hold the average dimension for a class, regressed value is the residual
"""
class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        class_ = class_.lower()
        return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map