
import torch
import os
import operator
from preprocess.data_preprocess import TestTransform
from postprocess.postprocessing import *
from datasets.KittiDatasets import KittiDatasets
from datasets.distributed_sampler import InferenceSampler
from datasets.dataset_utils import AspectRatioGroupedDataset
from torch.utils import data
from postprocess.evaluating import *
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import sys

class DefaultEvaluator:
    """
    Create a simple end-to-end evaluator with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.

    This evaluator takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Examples:

    .. code-block:: python

        pred = DefaultEvaluator(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, model, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        dataset_path = os.path.join(self.cfg.DATASET.PATH, 'training')
        self.model = model
        self.model.eval()
        self.device = torch.device(self.cfg.DEVICE if torch.cuda.is_available() else "cpu")
        checkpoint_file = os.path.join(root_path, self.cfg.DETECTOR.CHECKPOINT)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.transform = TestTransform(self.cfg.INTENSOR_SHAPE[0], self.cfg.DATASET.MEAN)
        dataset = KittiDatasets(dataset_path,
                                transform=self.transform)
        self.data_num = len(dataset)
        sampler = InferenceSampler(len(dataset))
        params = {'sampler': sampler,
                  'batch_sampler': None,
                  'collate_fn': operator.itemgetter(0),  # don't batch, but yield individual elements
                  'num_workers': 6}

        generator = data.DataLoader(dataset, **params)
        generator = AspectRatioGroupedDataset(generator, self.cfg.BATCH_SIZE)
        self.generator = iter(generator)
        self.model.to(self.device)

    def runing(self):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        cnt = 0
        for batch_data in self.generator:
            b_images = [d['image'] for d in batch_data]
            b_labels = [d['labels'] for d in batch_data]
            cnt += len(b_labels)
            b_images = torch.stack(b_images, 0)
            b_images = b_images.to(self.device)
            with torch.no_grad():
                predictions = self.model(b_images)
                processed_predictions = []
                for predictions_per_image in predictions:
                    r = detector_postprocess(predictions_per_image, 1, 1)
                    processed_predictions.append(r)
                tp, fp, fn = eval_tp_fp_fn(processed_predictions,b_labels, device=self.device)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            sys.stdout.write('\rEvaluating progress: {:.2f}'.format(100.*float(cnt)/self.data_num))
            sys.stdout.flush()
        precision = total_tp / max((total_tp + total_fp), 1)
        recall = total_tp / max((total_tp + total_fn), 1)
        f1_score = 2*precision*recall/(precision + recall)
        return f1_score, precision, recall
