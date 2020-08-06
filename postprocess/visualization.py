import numpy as np
import cv2
from utils.Utils import rotation_matrix
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

def draw_bboxes_to_image(image, bboxes, classes, label_map=None):
    height, width, channels = image.shape
    bboxes[:, 0] *= width
    bboxes[:, 2] *= width
    bboxes[:, 1] *= height
    bboxes[:, 3] *= height
    scale = 0.5
    offset = int(12 * scale)
    colormap = getColorMap()
    for i, bbox in enumerate(bboxes):
        color = colormap[classes[i]]
        label = '{}:{:.2f}'.format(label_map[classes[i]] if label_map is not None else classes[i], 1)
        cv2.putText(image, label,
                    (max(0, int(bbox[0]) - offset), max(0, int(bbox[1]) - offset)),
                    cv2.FONT_HERSHEY_COMPLEX, scale, color,
                    thickness=1)
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness=2)

    return image

def draw_instance_to_image(image, instance, label_map=None):
    bboxes = instance.get('pred_boxes').tensor.numpy()
    scores = instance.get('scores').numpy()
    pred_classes = instance.get('pred_classes').numpy()
    num = bboxes.shape[0]
    scale = 0.5
    offset = int(12 * scale)
    colormap = getColorMap()
    for i in range(num):
        color = colormap[pred_classes[i]]
        label = '{}:{:.2f}'.format(label_map[pred_classes[i]] if label_map is not None else pred_classes[i],
                               scores[i])

        cv2.putText(image, label,
                    (max(0, int(bboxes[i, 0])-offset), max(0, int(bboxes[i, 1])-offset)),
                    cv2.FONT_HERSHEY_COMPLEX, scale, color,
                    thickness=1)
        image = cv2.rectangle(image,
                              (int(bboxes[i, 0]), int(bboxes[i, 1])),
                              (int(bboxes[i, 2]), int(bboxes[i, 3])), color, thickness=2)

    return image


def plot_3d_box(img, proj_matrix, ry, dimension, center):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, proj_matrix=proj_matrix)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,proj_matrix=proj_matrix, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, proj_matrix)
        box_3d.append(point)

    #TODO put into loop
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)

    front_mark = np.array([[box_3d[0][0], box_3d[0][1]],
                           [box_3d[1][0], box_3d[1][1]],
                           [box_3d[3][0], box_3d[3][1]],
                           [box_3d[2][0], box_3d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    # cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 1)
    # cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, cv_colors.GREEN.value, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))