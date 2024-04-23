import torch
import numpy as np
import cv2
from numpy import random


def draw_boundary_boxes(detections, img):
    """*******************************************
        Functionality: Draw boundary boxes on received image
        Parameters: outputs(xy_wh,id,cls), img(in cv2 format), classes list
        Returns: img with boundary boxes
    **********************************************"""

    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        conf = detection["confidence"]
        object_label = detection["label"]
        txt_tobe_print = f'{conf} {object_label}'
        # txt_tobe_print = f'{tracker_id}'

        plot_one_box(bbox, img, label=txt_tobe_print, color=(255, 0, 0), line_thickness=1)
    return img


def plot_one_box(bbox, img, color=None, label=None, line_thickness=3):
    """*****************************
    Functionality : Draw boundary box
    ********************************"""

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        # tf = max(tl - 1, 1)  # font thickness
        tf = 1
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


