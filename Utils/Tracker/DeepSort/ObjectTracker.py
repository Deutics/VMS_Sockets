# Libraries
import time

from numpy import random

from .deep_sort import DeepSort
from .deep_sort.utils.parser import get_config
from .deep_sort.utils.utils import *

import cv2


class ObjectTracker:
    def __init__(self,
                 config_deepsort_path='utils/Tracker/DeepSort/deep_sort/configs/deep_sort.yaml',
                 obj_size=None,
                 use_gpu=True,
                 ):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort_path)

        self._tracker = DeepSort('osnet_x0_25',
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=use_gpu)

        self._colors = [random.randint(0, 255) for _ in range(3)]
        self._obj_size = obj_size

    def process_frame(self, frame, predictions_from_detector):
        """*************************************************************************
        Functionality:Takes a frame sends the frame to YoloDetector and gets its predictions
        apply strong sort functionality and show the tracked image
        Parameters:Takes a frame
        ****************************************************************************"""

        tracked_objects = []

        if len(predictions_from_detector):
            xy_whs, conf_rates, labels = self._remove_extra_detections(predictions_from_detector)

            if len(xy_whs):
                tracker_detections = self._tracker.update(xy_whs.cpu(), conf_rates.cpu(), labels.cpu(), frame)

                for i, detection in enumerate(tracker_detections):

                    temp = {
                        "bbox": detection[:4].tolist(),
                        "tracker_id": detection[4]
                    }
                    tracked_objects.append(temp)

        return tracked_objects

    def _remove_extra_detections(self, detections):
        """***********************************************
        Functionality: Converts detections from xy_xy format to xy_wh
        then removes all the extra detection that we not need
        Parameter: Detections in Tensor type
        Returns: Tensor(xy_wy), Tensor(confs), Tensor(clss)
        **************************************************"""
        boxes = xyxy2xywh(detections[:, 0:4])

        conf_rate = detections[:, 4:5]
        label = detections[:, 5:6]
        return boxes, conf_rate, label

    @property
    def tracker(self):
        return self._tracker

    @property
    def classes(self):
        return self._names_of_classes
