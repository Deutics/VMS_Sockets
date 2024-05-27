import time

import torch
import os

from .utils import download_model, get_indexes


class YoloDetector:
    def __init__(self, model_name="yolov5s.pt", conf_threshold=0.55,
                 iou_thresh=0.7,use_gpu=False, expected_objs=['car'], repository='ultralytics/yolov5',
                 obj_size = None):

        # model_directory_path = "utils\\ObjectDetectors\\Models\\Yolov5"
        model_directory_path = "C:\\shovalsc\\VMS_Sockets\\Utils\\ObjectDetector\\Models\\Yolov5"

        # download_model(model_path=model_directory_path, model_name=model_name)
        self._weights_file_path = os.path.join(model_directory_path, model_name)

        self._model = torch.hub.load(repository,
                                     'custom',
                                     self._weights_file_path,
                                     force_reload=False,
                                     verbose=False)
        self._model.conf = conf_threshold  # NMS confidence threshold
        self._model.iou = iou_thresh
        device = 0 if use_gpu and torch.cuda.is_available() else "cpu"
        print(device)

        self._model.to(device)  # specifying device type
        self._names = self._model.names
        self._model.classes = get_indexes(self._names, expected_classes=expected_objs)\
            if isinstance(expected_objs, list) else None

    @property
    def names(self):
        return self._names

    def process_frame(self, frame):
        results = self._model(frame)
        detections = results.xyxy[0]
        detected_objects = []

        for i in range(len(detections)):
            # print(detections[i])
            temp = {
                "label": detections["name"][i],
                "bbox": [int(detections["xmin"][i]), int(detections["ymin"][i]),
                         int(detections["xmax"][i]), int(detections["ymax"][i])],
                "confidence": detections["confidence"][i]
            }
            detected_objects.append(temp)

        return detected_objects

    def process_frame_for_tracker(self, frame):
        start_time = time.time()
        results = self._model(frame)
        detections = results.xyxy[0]
        print((time.time()-start_time)*1000)
        return detections
