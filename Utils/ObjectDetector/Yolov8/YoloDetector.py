from ultralytics import YOLO
import os
import torch

from .utils import *
from ..utils import *


class YoloDetector:
    def __init__(self, model_name="yolov8s.pt",
                 conf_threshold=0.45,
                 iou_thresh=0.7,
                 use_gpu=False,
                 expected_objs=None,
                 repository=None,
                 obj_size=None):

        model_directory_path = "C:\\shovalsc\\VMS_Sockets\\Utils\\ObjectDetector\\Models\\Yolov8"

        # download_model(model_path=model_directory_path, model_name=model_name)

        self._weights_file_path = os.path.join(model_directory_path, model_name)
        print("weights path", self._weights_file_path)
        self._model = YOLO(self._weights_file_path)

        self._conf_thresh = conf_threshold
        self._iou_thresh = iou_thresh
        self._device = 0 if use_gpu and torch.cuda.is_available() else "cpu"

        self._names = self._model.names
        # print(self._names)
        self._object_size_thresholds = self.calculate_size_bounds(obj_size)

        self._expected_objs = get_indexes(self._names, expected_classes=expected_objs)\
            if isinstance(expected_objs, list) else None

    def process_frame(self, frame):
        results = self._model.predict(source=frame, conf=self._conf_thresh, classes=self._expected_objs,
                                      device=self._device, verbose=False, iou=self._iou_thresh)
        detected_objects = []

        for result in results:
            for list_number in range(len(result)):
                boxes = result[list_number].boxes  # Boxes object for bbox outputs
                box = boxes[0]
                dimensions = box.xyxy.tolist()
                cls_index = box.cls.tolist()
                class_name = self._names[int(cls_index[0])]
                conf_score = round(box.conf.tolist()[0], 2)
                # conf_score = int(conf_score[0] * 100)
                dimensions = dimensions[0]
                dimensions = [int(dimensions) for dimensions in dimensions]
                if self._is_not_outlier_detection(detection=dimensions):
                    # storing values inside a list
                    detected_objects.append({
                        'label': class_name,
                        'bbox': dimensions,
                        'confidence': conf_score
                    })

        return detected_objects

    def process_frame_for_tracker(self, frame):
        results = self._model.predict(source=frame, conf=self._conf_thresh, classes=self._expected_objs,
                                      device=self._device, verbose=False, iou=self._iou_thresh)
        return results[0].boxes.data

    def process_frame_for_detector(self, frame):
        results = self._model.predict(source=frame, conf=self._conf_thresh, classes=self._expected_objs,
                                      device=self._device, verbose=False, iou=self._iou_thresh)

        return results[0].boxes.data

    def _is_not_outlier_detection(self, detection):
        box = xyxy2xywh(detection)

        if self._object_size_thresholds is not None:

            return (self._object_size_thresholds["min_width"] <= box[2] <= self._object_size_thresholds["max_width"])\
                and (self._object_size_thresholds["min_height"] <= box[3] <= self._object_size_thresholds["max_height"])

        return True

    @staticmethod
    def calculate_size_bounds(object_size):
        object_size_thresholds = None

        if object_size is not None:
            precision_factor = object_size["Precision"] * 0.01

            max_width = object_size["Width"] + (object_size["Width"] * precision_factor)
            max_height = object_size["Height"] + (object_size["Height"] * precision_factor)

            min_width = object_size["Width"] - (object_size["Width"] * precision_factor)
            min_height = object_size["Height"] - (object_size["Height"] * precision_factor)

            object_size_thresholds = {"max_width": max_width,
                                      "max_height": max_height,
                                      "min_width": min_width,
                                      "min_height": min_height}

        return object_size_thresholds

    @property
    def names(self):
        return self._names
