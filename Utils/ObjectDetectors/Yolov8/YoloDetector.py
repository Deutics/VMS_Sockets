import time

from ultralytics import YOLO
import os
import torch

from .utils import *
from ..utils import *


class YoloDetector:
    def __init__(self, model_name="yolov8s.pt",
                 conf_threshold=0.,
                 iou_thresh=0.7,
                 use_gpu=False,
                 expected_objs=None,
                 repository=None,
                 obj_size=None):
        model_directory_path = "C:\\shovalsc\\VMS_Sockets_2.0\\Utils\\ObjectDetectors\\Models\\Yolov8"

        self._weights_file_path = os.path.join(model_directory_path, model_name)
        self._model = YOLO(self._weights_file_path)

        self._conf_thresh = conf_threshold
        self._iou_thresh = iou_thresh
        self._device = 0 if use_gpu and torch.cuda.is_available() else "cpu"

        # print("device: ", self._device)

        self._names = self._model.names

        self._expected_objs = get_indexes(self._names, expected_classes=expected_objs)\
            if isinstance(expected_objs, list) else None

    def process_frame_for_tracker(self, frame, expected_objects):
        results = self._model.predict(source=frame, conf=self._conf_thresh,classes=expected_objects,
                                      device=self._device, verbose=False, iou=self._iou_thresh)
        return results[0].boxes.data

    @property
    def names(self):
        return self._names
