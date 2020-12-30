import numpy as np
import cv2
# from utils.Utils import rotation_matrix
from postprocess.postprocessing import *
from enum import Enum

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)
    DINGXIANG = (204, 164, 227)

KITTI_COLOR_MAP = (
cv_colors.RED.value,
cv_colors.GREEN.value,
cv_colors.BLUE.value,
cv_colors.PURPLE.value,
cv_colors.ORANGE.value,
cv_colors.MINT.value,
cv_colors.YELLOW.value,
cv_colors.DINGXIANG.value
)

def getColorMap():
    colormap = [[255, 255, 255]]
    for i in range(3*9 - 1):
        if i % 9 == 0:
            continue
        k = i // 9
        m = i % 9
        color = [255, 255, 255]
        color[k] = (color[k] >> m)
        colormap.append(color)
    return colormap

def cv_draw_bboxes_2d(image, bboxes_2d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes = bboxes_2d.get_field('bbox').cpu().numpy()
    classes = bboxes_2d.get_field('class').cpu().numpy().astype(np.int)
    scale = 0.5
    offset = int(12 * scale)
    for cls, bbox in zip(classes, bboxes):
        color = color_map[cls]
        label = '{}:{:.2f}'.format(label_map[cls] if label_map is not None else cls, 1)
        cv2.putText(image, label,
                    (max(0, int(bbox[0]) - offset), max(0, int(bbox[1]) - offset)),
                    cv2.FONT_HERSHEY_COMPLEX, scale, color,
                    thickness=1)
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color)

    return image

def cv_draw_bboxes_3d(img, bboxes_3d, label_map=None, color_map=KITTI_COLOR_MAP):
    classes = bboxes_3d.get_field('class').cpu().numpy().astype(np.int)
    scores = bboxes_3d.get_field('score').cpu().numpy()
    locations = bboxes_3d.get_field('location').cpu().numpy()
    Rys = bboxes_3d.get_field('Ry').cpu().numpy()
    dimensions = bboxes_3d.get_field('dimension').cpu().numpy()
    K = bboxes_3d.get_field('K').cpu().numpy()

    for cls, loc, Ry, dim, score in zip(classes, locations, Rys, dimensions, scores):
        label = label_map[cls] if label_map is not None else cls
        cv_draw_bbox_3d(img, K, Ry, dim, loc, label, score, color_map[cls])

def cv_draw_bbox_3d(img, proj_matrix, ry, dimension, center, cls, score, color):

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, proj_matrix)
        box_3d.append(point)

    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), color, 1)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), color, 1)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), color, 1)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), color, 1)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), color, 1)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), color, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), color, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), color, 1)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), color, 1)

    front_mark = np.array([[box_3d[0][0], box_3d[0][1]],
                           [box_3d[1][0], box_3d[1][1]],
                           [box_3d[3][0], box_3d[3][1]],
                           [box_3d[2][0], box_3d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))