import importlib
import cv2
from .utils import *

detector_classes = {"Yolov3": "YoloDetector",
                    "MobileNetSSD": "MobileNetSSD",
                    "Yolov5": "YoloDetector",
                    "Yolov7": "YoloDetector",
                    "Yolov8": "YoloDetector",
                    "Yolov9": "YoloDetector"}


def get_detector_instance(detector,
                          model_name,
                          conf_threshold,
                          iou_threshold,
                          use_gpu,
                          expected_objects,
                          repository,
                          obj_size):

    detector = detector.capitalize()
    full_module_name = f"Utils.ObjectDetector.{detector}.{detector_classes[detector]}"
    module = importlib.import_module(full_module_name)

    detector_instance = getattr(module, detector_classes[detector])

    return detector_instance(model_name=model_name,
                             conf_threshold=conf_threshold,
                             use_gpu=use_gpu,
                             expected_objs=expected_objects,
                             iou_thresh=iou_threshold,
                             repository=repository,
                             obj_size=obj_size)


class ObjectDetector:
    def __init__(self, detector="Yolov8",
                 model_name="yolov8l.pt",
                 use_gpu=True,
                 conf_threshold=0.5,
                 iou_threshold=0.7,
                 expected_objects=None,
                 repository='ultralytics/yolov5',
                 obj_size=None):

        self._model = get_detector_instance(detector=detector,
                                            model_name=model_name,
                                            conf_threshold=conf_threshold,
                                            use_gpu=use_gpu,
                                            expected_objects=expected_objects,
                                            iou_threshold=iou_threshold,
                                            repository=repository,
                                            obj_size=obj_size)

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

            frame = cv2.resize(frame, (720, 480))      # landscape
            # frame = cv2.resize(frame, (480, 720))      # portrait
            detections = self.process_frame(frame)

            # print(detections)

            if len(detections):
                draw_boundary_boxes(detections=detections, img=frame)

            cv2.imshow(streaming_source, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        return self._model.process_frame(frame)

    def process_frame_for_tracker(self, frame):
        return self._model.process_frame_for_tracker(frame)

    def names(self):
        return self._model.names
