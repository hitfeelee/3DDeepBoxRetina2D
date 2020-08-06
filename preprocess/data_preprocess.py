from .transform import *


class TrainAugmentation:
    def __init__(self, size, mean=0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
#            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            ToTensor(),
            ToNCHW(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToBBoxes()
        ])

    def __call__(self, img, bboxes=None, labels=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, bboxes=bboxes, labels=labels)


class TestTransform:
    def __init__(self, size, mean=0.0):
        self.transform = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            ToTensor(),
            ToNCHW(),
            ToBBoxes()
        ])

    def __call__(self, image, bboxes=None, labels=None):
        return self.transform(image, bboxes=bboxes, labels=labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0):
        self.transform = Compose([
            ConvertFromInts(),
            Resize(size),
            SubtractMeans(mean),
            ToTensor(),
            ToNCHW()
        ])

    def __call__(self, image, bboxes=None, labels=None):
        image, _, _ = self.transform(image, bboxes=bboxes, labels=labels)
        return image
