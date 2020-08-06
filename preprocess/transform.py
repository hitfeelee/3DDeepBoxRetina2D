# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from Archs_2D.BBox import BBoxes

def intersect(box_a, box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of bboxes.  The jaccard overlap
    is simply the intersection over union of two bboxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding bboxes, Shape: [num_bboxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes=bboxes, labels=labels)
        return img, bboxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, bboxes=None, labels=None):
        return self.lambd(img, bboxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, bboxes=None, labels=None):
        return image.astype(np.float32), bboxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, bboxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), bboxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, bboxes=None, labels=None):
        height, width, channels = image.shape
        bboxes[:, 0] *= width
        bboxes[:, 2] *= width
        bboxes[:, 1] *= height
        bboxes[:, 3] *= height

        return image, bboxes, labels


class ToPercentCoords(object):
    def __call__(self, image, bboxes=None, labels=None):
        height, width, channels = image.shape
        bboxes[:, 0::2] /= width
        bboxes[:, 1::2] /= height

        return image, bboxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, bboxes=None, labels=None):
        if isinstance(self.size, (tuple, list)) and len(self.size) == 2:
            if isinstance(image, list):
                image = [cv2.resize(src=src, dsize=self.size, interpolation=cv2.INTER_CUBIC) for src in image]
            else:
                image = cv2.resize(src=image, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        else:
            if isinstance(image, list):
                image = [cv2.resize(src=src, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC) for src in image]
            else:
                image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return image, bboxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, bboxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, bboxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, bboxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, bboxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, bboxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, bboxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, bboxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, bboxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), bboxes, labels


# class ToTensor(object):
#     def __call__(self, cvimage, bboxes=None, labels=None):
#         return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), bboxes, labels

class ToTorchTensor(object):
    def __init__(self):
        self.toTensor = transforms.ToTensor()
    def __call__(self, image, bboxes=None, labels=None):
        if bboxes is not None:
            if not torch.is_tensor(bboxes):
                bboxes = torch.from_numpy(bboxes.astype(np.float32))
        if isinstance(image, list):
            image = [self.toTensor(src) for src in image]
        else:
            image = self.toTensor(image)
        return image, bboxes, labels

class ToTensor(object):
    def __init__(self):
        self.trans = transforms.ToTensor()
    def __call__(self, cvimage, bboxes=None, labels=None):
        timg = torch.from_numpy(cvimage.astype(np.float32))
        # timg = self.trans(cvimage)
        if bboxes is not None:
            bboxes = torch.from_numpy(bboxes.astype(np.float32))
        if labels is not None:
            labels = torch.from_numpy(labels.astype(np.long))

        return timg, bboxes, labels

class ToNCHW(object):
    def __call__(self, image, bboxes=None, labels=None):
        return image.permute(2, 0, 1), bboxes, labels

class ToBBoxes(object):
    def __call__(self, image, bboxes=None, labels=None):
        return image, BBoxes(bboxes), labels


class RandomCrop(object):
    def __init__(self, rate=0.3):
        self._rate = rate

    def __call__(self, image, bboxes=None, labels=None):
        height, width, _ = image.shape
        if isinstance(bboxes, (tuple, list)):
            bboxes = np.array(bboxes, dtype=np.float32)
        elif torch.is_tensor(bboxes):
            bboxes = bboxes.numpy().astype(dtype=np.float32)
        assert(bboxes.ndim == 1)
        bboxes[0::2] = np.clip(bboxes[0::2], a_min=0, a_max=width)
        bboxes[1::2] = np.clip(bboxes[1::2], a_min=0, a_max=height)
        if random.randint(2):
            # crop image
            x_range = (bboxes[2] - bboxes[0])*self._rate
            y_range = (bboxes[3] - bboxes[1])*self._rate
            det_bboxes = np.random.random_sample(bboxes.shape)
            det_bboxes[0::2] = det_bboxes[0::2] * x_range * 2 - x_range
            det_bboxes[1::2] = det_bboxes[1::2] * y_range * 2 - y_range
            bboxes += det_bboxes
            bboxes[0::2] = np.clip(bboxes[0::2], a_min=0, a_max=width)
            bboxes[1::2] = np.clip(bboxes[1::2], a_min=0, a_max=height)
        crop = image[int(bboxes[1]):(int(bboxes[3]) + 1), int(bboxes[0]):(int(bboxes[2]) + 1)]
        return crop, bboxes, labels

class Crop(object):
    def __call__(self, image, bboxes=None, labels=None):
        height, width, _ = image.shape
        if isinstance(bboxes, (tuple, list)):
            bboxes = np.array(bboxes, dtype=np.float32)
        elif torch.is_tensor(bboxes):
            bboxes = bboxes.numpy().astype(dtype=np.float32)
        assert(bboxes.ndim == 2)#(num, 4)
        num, _ = bboxes.shape
        crop = []
        for i in range(num):
            bbox = bboxes[i]
            bbox[0::2] = np.clip(bbox[0::2], a_min=0, a_max=width)
            bbox[1::2] = np.clip(bbox[1::2], a_min=0, a_max=height)
            crop.append(image[int(bbox[1]):(int(bbox[3]) + 1), int(bbox[0]):(int(bbox[2]) + 1)])
        return crop, bboxes, labels

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        bboxes (Tensor): the original bounding bboxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, bboxes, labels)
            img (Image): the cropped image
            bboxes (Tensor): the adjusted bounding bboxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, bboxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, bboxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt bboxes
                overlap = jaccard_numpy(bboxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0

                # mask in all gt bboxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt bboxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid bboxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt bboxes
                current_bboxes = bboxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_bboxes[:, :2] = np.maximum(current_bboxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_bboxes[:, :2] -= rect[:2]

                current_bboxes[:, 2:] = np.minimum(current_bboxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_bboxes[:, 2:] -= rect[:2]

                return current_image, current_bboxes, current_labels


# class Expand(object):
#     def __init__(self, mean):
#         self.mean = mean
#
#     def __call__(self, image, bboxes, labels):
#         if random.randint(2):
#             return image, bboxes, labels
#
#         height, width, depth = image.shape
#         ratio = random.uniform(1, 4)
#         left = random.uniform(0, width*ratio - width)
#         top = random.uniform(0, height*ratio - height)
#
#         expand_image = np.zeros(
#             (int(height*ratio), int(width*ratio), depth),
#             dtype=image.dtype)
#         expand_image[:, :, :] = self.mean
#         expand_image[int(top):int(top + height),
#                      int(left):int(left + width)] = image
#         image = expand_image
#
#         bboxes = bboxes.copy()
#         bboxes[:, :2] += (int(left), int(top))
#         bboxes[:, 2:] += (int(left), int(top))
#
#         return image, bboxes, labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, bboxes=None, labels=None):
        if random.randint(3) == 0:
            return image, bboxes, labels
        elif random.randint(3) == 1:
            height, width, depth = image.shape
            ratio = random.uniform(1, 3)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
                         int(left):int(left + width)] = image
            image = expand_image

            bboxes = bboxes.copy()
            bboxes[:, :2] += (int(left), int(top))
            bboxes[:, 2:] += (int(left), int(top))

            return image, bboxes, labels
        else:
            height, width, depth = image.shape
            ratio = random.uniform(1./3, 1.)
            left = random.uniform(0, width - width * ratio)
            top = random.uniform(0, height - height * ratio)
            expand_image = image.copy()
            expand_image = expand_image[int(top):int(top + height*ratio),
                    int(left):int(left + width*ratio)]
            expand_h, expand_w, _ = expand_image.shape
            win = np.array([[int(left), int(top), expand_w, expand_h]])
            expand_bboxes = bboxes.copy()
            bottom_right = np.where(expand_bboxes[:, 2:] < win[:, 2:], expand_bboxes[:, 2:], win[:, 2:])
            top_left = np.where(expand_bboxes[:, :2] > win[:, :2], expand_bboxes[:, :2], win[:, :2])
            width_height = bottom_right - top_left
            width_height = width_height.clip(min=0)
            inter = np.prod(width_height, axis=1)
            index = inter > 0
            expand_bboxes = expand_bboxes[index]
            expand_labels = labels[index]
            expand_bboxes[:, :2] -= (int(left), int(top))
            expand_bboxes[:, 2:] -= (int(left), int(top))
            expand_bboxes[:, 0::2] = expand_bboxes[:, 0::2].clip(min=0, max=int(expand_w))
            expand_bboxes[:, 1::2] = expand_bboxes[:, 1::2].clip(min=0, max=int(expand_h))
            if expand_labels.shape[0] > 0:
                return expand_image, expand_bboxes, expand_labels
            else:
                return image, bboxes, labels



class RandomMirror(object):
    def __call__(self, image, bboxes=None, labels=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = width - bboxes[:, 2::-2]
        return image, bboxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, bboxes=None, labels=None):
        im = image.copy()
        im, bboxes, labels = self.rand_brightness(im, bboxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, bboxes, labels = distort(im, bboxes, labels)
        return self.rand_light_noise(im, bboxes, labels)


class Normalize(object):
    def __init__(self, mean=[], std=[]):
        self._normalize = transforms.Normalize(mean, std)

    def __call__(self, image, bboxes=None, labels=None):
        if isinstance(image, list):
            image = [self._normalize(src) for src in image]
        else:
            image = self._normalize(image)
        return image, bboxes, labels