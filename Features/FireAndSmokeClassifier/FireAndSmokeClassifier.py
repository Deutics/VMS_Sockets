import json
import time
import numpy as np
from datetime import datetime


# Tracker
from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker


from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT
from Utils.structures.camera_polygon import CameraPolygon
from Utils.structures.detection_cam_config import DetectionCameraConfig
from Utils.structures.camera_info import CameraInfo
from Utils.MemoryMappedFile.MemoryMappedFile import MemoryMappedFileHandler

from Features.utils import *


class FireAndSmokeClassifier:
    def __init__(self, data_dict
                 ):

        self.bbox = None
        self.mmp_file = [None] * 5
        self.polygon_data = {}

        self._object_tracker = ObjectTracker(use_gpu=True)
        self._expected_objs = []

        self.camera_id = data_dict['cameraId']
        self.video_index = data_dict['videoIndex']
        self.algo_type = data_dict['algoType']
        self.video_width = data_dict['videoWidth']
        self.video_height = data_dict['videoHeight']
        self.data_size = data_dict['dataSize']
        self.send_always = data_dict['sendAlways']

        self.first_frame = 1
        self.camera_polygons = []
        self.total_camera_polygons = -1
        self.frame_number = 0
        self._last_detection_time = time.time()
        self.detection_cam_config = None
        self.source_frame_number = 0
        self.running = True

        self.initialize_features(data_dict)

        self.initialize_bbox()

    def initialize_bbox(self):
        xmin = (self.video_width - 100) // 2
        ymin = (self.video_height - 50) // 2
        xmax = xmin + 100
        ymax = ymin + 50
        self.bbox = [xmin, ymin, xmax, ymax]

    def process_frame(self, frame, region_of_interest, detections_from_detector):
        tracked_objects = self._object_tracker.process_frame(frame=frame,
                                                             predictions_from_detector=detections_from_detector)

        intruded_objects = None
        if len(tracked_objects):
            # plot_one_box(self.bbox, frame, label="fire/smoke", color=(0, 0, 255), line_thickness=1)
            self._last_detection_time = time.time()
            intruded_objects = self._send_warning()
        else:
            if (time.time() - self._last_detection_time) >= 1:
                intruded_objects = self.send_clearance_message()

        return intruded_objects

    def _send_warning(self):
        intruded_object_data = ALGO_DETECTION_OBJECT_DATA(X=int(self.bbox[0]), Y=int(self.bbox[1]),
                                                          Width=int(self.bbox[2] - self.bbox[0]),
                                                          Height=int(self.bbox[3] - self.bbox[1]),
                                                          CountUpTime=0, ObjectType="fire/smoke",
                                                          frameNum=self.frame_number, ID=0, polygonID=1,
                                                          DetectionPercentage=100)

        return self.create_structure([intruded_object_data])

    def region_of_interest(self, frame):
        return frame

    def create_structure(self, intruded_object_data):

        intruded_object_by_type = ALGO_DETECTION_OBJECTS_BY_TYPE(objectCount=len(intruded_object_data),
                                                                 alarmSet=bool(len(intruded_object_data)),
                                                                 objectsType="fire/smoke",
                                                                 algObject=intruded_object_data)
        intruded_object_by_type = [intruded_object_by_type]

        intruded_objects = ALGO_DETECTION_OBJECT(cameraId=self.camera_id,
                                                 totalObjectCount=len(intruded_object_by_type),
                                                 videoWidth=self.video_width,
                                                 videoHeight=self.video_height,
                                                 dateTime=datetime.now(),
                                                 AlgoType=self.algo_type,
                                                 videoCounter=1,
                                                 detectionCameraConfig=self.detection_cam_config,
                                                 algoObject=intruded_object_by_type
                                                 )
        detection = intruded_objects.toJSON().encode('utf-8')
        # print(detection)
        return detection

    def initialize_cam_config(self):
        camera_info = CameraInfo(polygon_available=True,
                                 video_counter=1,
                                 video_width=self.video_width,
                                 video_height=self.video_height)

        return DetectionCameraConfig(cameraPolygon=self.camera_polygons, cameraInfo=camera_info)

    def fetch_frame_buffer(self):
        if self.mmp_file[0] is None:
            mm_file_name = f"ShovalSCMMap_vid_{self.camera_id}"
            self.mmp_file[0] = MemoryMappedFileHandler(file_name=mm_file_name,
                                                       video_width=self.video_width,
                                                       video_height=self.video_height)

        buf = self.mmp_file[0].read_content()

        image_array = None
        if len(buf):
            image_array = np.frombuffer(buf, dtype=np.uint8).reshape(self.video_height, self.video_width, 3)

        return image_array

    def initialize_features(self, data_dict):
        try:
            # self.camera_polygons.append(CameraPolygon("temp"))
            # if len()
            for polygon_data in data_dict["CameraPolygon"]:
                camera_polygon = CameraPolygon(polygon_data)

                self.camera_polygons.append(camera_polygon)
                detection_types = json.loads(polygon_data["DetectionAndAlertCount"])
                temp_detections = []

                for detection_type in detection_types:
                    temp_detections.append(detection_type["AIDetectiontype"])
                    if detection_type["AIDetectiontype"] not in self._expected_objs:
                        self._expected_objs.append(detection_type["AIDetectiontype"])

                self.polygon_data[polygon_data["PolygonId"]] = temp_detections

        except (TypeError, KeyError) as e:
            print(f"Error mapping polygons: {e}")

        self.detection_cam_config = self.initialize_cam_config()

    def send_clearance_message(self):
        self._last_detection_time = time.time()
        print("sending clearnace")
        return self.create_structure([])


