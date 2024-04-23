# Libraries
from numpy import random

from .deep_sort import DeepSort
from .deep_sort.utils.parser import get_config
from .deep_sort.utils.utils import *

# Object Detector
from ...ObjectDetector.ObjectDetector import ObjectDetector
import cv2


class ObjectTracker:
    def __init__(self, detector="Yolov8",
                 detector_model="yolov8s.pt",
                 config_deepsort_path='utils/Tracker/DeepSort/deep_sort/configs/deep_sort.yaml',
                 obj_size=None,
                 conf_thresh=0.25,
                 iou_thresh = 0.7,
                 use_gpu=False,
                 expected_objs=None,
                 repository ='ultralytics/yolov5'
                 ):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort_path)
        self.object_detector = ObjectDetector(detector=detector,
                                              model_name=detector_model,
                                              use_gpu=use_gpu,
                                              conf_threshold=conf_thresh,
                                              expected_objects=expected_objs,
                                              repository='ultralytics/yolov5')

        self._tracker = DeepSort('osnet_x0_25',
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        self._colors = [random.randint(0, 255) for _ in range(3)]
        self._names_of_classes = self.object_detector.names()
        self._obj_size = obj_size

    def process_video(self, streaming_source):
        """******************************
        Functionality: read the frame of video, and send it to function process image
        Parameters: path of video
        Returns: None
        *********************************"""
        cap = cv2.VideoCapture(streaming_source)

        while True:
            is_frame, frame = cap.read()
            if not is_frame:
                break
            tracked_objects = self.process_frame(frame)
            if len(tracked_objects):
                frame = draw_boundary_boxes(tracked_objects,frame)
            cv2.imshow(streaming_source, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """*************************************************************************
        Functionality:Takes a frame sends the frame to YoloDetector and gets its predictions
        apply strong sort functionality and show the tracked image
        Parameters:Takes a frame
        ****************************************************************************"""
        # Yolo predictions
        predictions_from_detector = self.object_detector.process_frame_for_tracker(frame)

        tracked_objects = []

        if len(predictions_from_detector):
            xy_whs, conf_rates, labels = self._remove_extra_detections(predictions_from_detector)

            if len(xy_whs):
                tracker_detections = self._tracker.update(xy_whs.cpu(), conf_rates.cpu(), labels.cpu(), frame)

                for i, detection in enumerate(tracker_detections):

                    temp = {
                        "label": self._names_of_classes[int(detection[5])],
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

        # if self._obj_size is not None:
        #     # calculating max/min height and width we need
        #     max_width = self._obj_size["Width"] + (self._obj_size["Width"] * (self._obj_size["Precision"] * 0.01))
        #     max_height = self._obj_size["Height"] + (self._obj_size["Height"] * (self._obj_size["Precision"] * 0.01))
        #     min_width = self._obj_size["Width"] - (self._obj_size["Width"] * (self._obj_size["Precision"] * 0.01))
        #     min_height = self._obj_size["Height"] - (self._obj_size["Height"] * (self._obj_size["Precision"] * 0.01))
        #
        #     xy_wh = []
        #     conf_rate = []
        #     label = []
        #
        #     # removing extra boxes
        #     for i, detection in enumerate(boxes):
        #         if (max_width >= detection[2] >= min_width) and (max_height >= detection[3] >= min_height):
        #             xy_wh.append([float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])])
        #             conf_rate.append(detections[i, 4])
        #             label.append(detections[i, 5])
        #
        #     return torch.tensor(xy_wh, device='cuda:0'), torch.tensor(conf_rate, device='cuda:0'),\
        #         torch.tensor(label, device='cuda:0')

        conf_rate = detections[:, 4:5]
        label = detections[:, 5:6]
        return boxes, conf_rate, label

    @property
    def tracker(self):
        return self._tracker

    @property
    def classes(self):
        return self._names_of_classes
