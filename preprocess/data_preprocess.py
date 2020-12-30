from .transforms import *


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
            RandomAffine(self.mean),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            ToTensor(),
            ToNCHW(),
            ToBBoxes()
        ])

    def __call__(self, img, targets=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.augment(img, targets=targets)


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

    def __call__(self, img, targets=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.transform(img, targets=targets)


class PredictionTransform:
    def __init__(self, size, mean=0.0):
        self.transform = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            ToTensor(),
            ToNCHW()
        ])

    def __call__(self, img, targets=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            targets: ground truth labels , type Paramlist.
        """
        return self.transform(img, targets=targets)
