from .transform import *


class TrainAugmentation:
    def __init__(self, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            RandomCrop(rate=0.01),
            Resize(self.size),
            ToTorchTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, bboxes=None, labels=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, bboxes=bboxes, labels=labels)

class PredictionTransform:
    def __init__(self, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            Crop(),
            Resize(self.size),
            ToTorchTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, bboxes=None, labels=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, bboxes=bboxes, labels=labels)