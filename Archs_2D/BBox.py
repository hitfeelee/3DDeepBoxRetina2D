# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union
import torch

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


__all__ = ["BBoxMode","BBoxes", "pairwise_iou","matched_boxlist_iou", "Box2BoxTransform", "Box2BoxTransformRotated"]

@unique
class BBoxMode(Enum):
    """
    Enum of different ways to represent a box.

    Attributes:

        XYXY_ABS: (x0, y0, x1, y1) in absolute floating points coordinates.
            The coordinates in range [0, width or height].
        XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
        XYXY_REL: (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
        XYWH_REL: (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
        XYWHA_ABS: (xc, yc, w, h, a) in absolute floating points coordinates.
            (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3
    XYWHA_ABS = 4

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BBoxMode", to_mode: "BBoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BBoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BBoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            arr = torch.from_numpy(np.asarray(box)).clone()  # avoid modifying the input box

        assert to_mode.value not in [
            BBoxMode.XYXY_REL,
            BBoxMode.XYWH_REL,
        ] and from_mode.value not in [
            BBoxMode.XYXY_REL,
            BBoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BBoxMode.XYWHA_ABS and to_mode == BBoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        else:
            if to_mode == BBoxMode.XYXY_ABS and from_mode == BBoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BBoxMode.XYXY_ABS and to_mode == BBoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BBoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class BBoxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4.
    """

    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "BBoxes":
        """
        Clone the BBoxes.

        Returns:
            BBoxes
        """
        return BBoxes(self.tensor.clone())

    def to(self, device: str) -> "BBoxes":
        return BBoxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BBoxes":
        """
        Returns:
            BBoxes: Create a new :class:`BBoxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `BBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned BBoxes might share storage with this BBoxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on BBoxes with {} failed to return a matrix!".format(item)
        return BBoxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "BBoxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): BBoxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @staticmethod
    def cat(bboxes_list: List["BBoxes"]) -> "BBoxes":
        """
        Concatenates a list of BBoxes into a single BBoxes

        Arguments:
            bboxes_list (list[BBoxes])

        Returns:
            BBoxes: the concatenated BBoxes
        """
        assert isinstance(bboxes_list, (list, tuple))
        assert len(bboxes_list) > 0
        assert all(isinstance(box, BBoxes) for box in bboxes_list)

        cat_bboxes = type(bboxes_list[0])(torch.cat([b.tensor for b in bboxes_list], dim=0))
        return cat_bboxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(bboxes1: BBoxes, bboxes2: BBoxes) -> torch.Tensor:
    """
    Given two lists of bboxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of bboxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        bboxes1,bboxes2 (BBoxes): two `BBoxes`. Contains N & M bboxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = bboxes1.area()
    area2 = bboxes2.area()

    bboxes1, bboxes2 = bboxes1.tensor, bboxes2.tensor

    width_height = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:]) - torch.max(
        bboxes1[:, None, :2], bboxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(bboxes1: BBoxes, bboxes2: BBoxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    bboxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Arguments:
        bboxes1: (BBoxes) bounding boxes, sized [N,4].
        bboxes2: (BBoxes) bounding boxes, sized [N,4].
    Returns:
        (tensor) iou, sized [N].
    """
    assert len(bboxes1) == len(bboxes2), (
        "boxlists should have the same"
        "number of entries, got {}, {}".format(len(bboxes1), len(bboxes2))
    )
    area1 = bboxes1.area()  # [N]
    area2 = bboxes2.area()  # [N]
    box1, box2 = bboxes1.tensor, bboxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes


class Box2BoxTransformRotated(object):
    """
    The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
    by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
    by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
    and rotate a box's angle by da (radians).
    Note: angles of deltas are in radians while angles of boxes are in degrees.
    """

    def __init__(self, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
        """
        Args:
            weights (5-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh, da) deltas. These are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
            target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(src_boxes, dim=1)

        target_ctr_x, target_ctr_y, target_widths, target_heights, target_angles = torch.unbind(
            target_boxes, dim=1
        )

        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        # Angles of deltas are in radians while angles of boxes are in degrees.
        # the conversion to radians serve as a way to normalize the values
        da = target_angles - src_angles
        da = (da + 180.0) % 360.0 - 180.0  # make it in [-180, 180)
        da *= wa * math.pi / 180.0

        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, 5).
                deltas[i] represents box transformation for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        """
        assert deltas.shape[1] == 5 and boxes.shape[1] == 5
        assert torch.isfinite(deltas).all().item(), "Box regression deltas become infinite or NaN!"

        boxes = boxes.to(deltas.dtype)

        ctr_x, ctr_y, widths, heights, angles = torch.unbind(boxes, dim=1)
        wx, wy, ww, wh, wa = self.weights
        dx, dy, dw, dh, da = torch.unbind(deltas, dim=1)

        dx.div_(wx)
        dy.div_(wy)
        dw.div_(ww)
        dh.div_(wh)
        da.div_(wa)

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = dx * widths + ctr_x  # x_ctr
        pred_boxes[:, 1] = dy * heights + ctr_y  # y_ctr
        pred_boxes[:, 2] = torch.exp(dw) * widths  # width
        pred_boxes[:, 3] = torch.exp(dh) * heights  # height

        # Following original RRPN implementation,
        # angles of deltas are in radians while angles of boxes are in degrees.
        pred_angle = da * 180.0 / math.pi + angles
        pred_angle = (pred_angle + 180.0) % 360.0 - 180.0  # make it in [-180, 180)

        pred_boxes[:, 4] = pred_angle

        return pred_boxes
