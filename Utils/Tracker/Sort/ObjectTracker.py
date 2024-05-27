import time

from .sort.sort import *
from .sort.utils import *
import cv2


class ObjectTracker:
    def __init__(self,
                 obj_size=None,
                 use_gpu=True,):

        self._tracker = Sort()

    def process_frame(self, frame, predictions_from_detector):
        tracker_detections = []
        updated_detections = self.transform_detections(predictions_from_detector)
        tracker_detections = self._tracker.update(np.array(updated_detections))

        return tracker_detections

    @staticmethod
    def transform_detections(detections):
        detections = detections.cpu().tolist()
        transformed_detections = []
        for i, detection in enumerate(detections):
            transformed_detections.append(detection)
        return transformed_detections

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
    def tracker(self):
        return self._tracker
