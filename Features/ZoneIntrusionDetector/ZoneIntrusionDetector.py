import json
import time
import socket
import numpy as np
from datetime import datetime

# Tracker
from Utils.Tracker.Sort.ObjectTracker import ObjectTracker

from Utils.Zone.TimeSpecificZone import TimeSpecificZone

from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT
from Utils.structures.camera_polygon import CameraPolygon
from Utils.structures.detection_cam_config import DetectionCameraConfig
from Utils.structures.camera_info import CameraInfo
from Utils.MemoryMappedFile.MemoryMappedFile import MemoryMappedFileHandler

from Features.utils import *


class ZoneIntrusionDetector:

    def __init__(self, data_dict,
                 zone_dwelling_time_in_seconds=0,
                 ):

        self.mmp_file = [None] * 5  # array of 5
        self._expected_objs = []
        self._zone_dwelling_time_in_seconds = zone_dwelling_time_in_seconds

        self.polygon_zone = None
        self.polygon_data = {}

        self._object_tracker = ObjectTracker(use_gpu=True)

        self.region_min_x, self.region_min_y, self.region_max_x, self.region_max_y = 0, 0, 0, 0
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

    def process_frame(self, frame, region_of_interest, detections_from_detector):
        tracked_objects = self._object_tracker.process_frame(frame=region_of_interest,
                                                             predictions_from_detector=detections_from_detector)

        intruded_objects = None
        if len(tracked_objects):
            intruded_objects = self._check_zone_intrusion(frame)

        return intruded_objects

    def region_of_interest(self, frame):
        region_of_interest = cv2.bitwise_and(frame, frame, mask=self.polygon_zone.polygon_binary_mask)

        if self.total_camera_polygons == 1:
            region_of_interest = region_of_interest[self.region_min_y:self.region_max_y,
                                                    self.region_min_x:self.region_max_x]

        return region_of_interest

    def set_polygon_data(self):
        self.total_camera_polygons = len(self.camera_polygons)
        points = self.polygon_zone.zone_coordinates[0]
        if self.total_camera_polygons == 1:
            self.region_min_x, self.region_min_y = max(0, points.min(axis=0)[0]), max(0, points.min(axis=0)[1])
            self.region_max_x, self.region_max_y = min(self.video_height, points.max(axis=0)[0]), points.max(axis=0)[1]

        self.polygon_zone.create_polygon_mask((self.video_height, self.video_width))

    def draw_polygon(self, frame):
        for polygon in self.polygon_zone.zone_coordinates:
            pts = polygon.reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        return frame

    def _check_zone_intrusion(self, frame):
        """*********************************
        Functionality: Check if any of detected objects has intruded the defined zone
        Parameter: None
        Return: None
        ************************************"""

        tracks = self._object_tracker.tracker.get_tracks()
        intruded_objects = {}

        for i, track in enumerate(tracks):
            # if track.is_missed or not len(track.get_state()[0]):
            #     continue

            bbox = track.get_state()[0]

            track_id = track.track_id + 1

            # label = self._object_tracker.classes[track.class_id.item()]

            if self.total_camera_polygons == 1:
                xmin, ymin, xmax, ymax = bbox

                bbox = [xmin + self.region_min_x, ymin + self.region_min_y,
                        xmax + self.region_min_x, ymax + self.region_min_y]

            x, y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

            polygon_id = self.polygon_zone.polygon_binary_mask[y][x]

            # if label in self.polygon_data[polygon_id]:
            # txt_tobe_print = f'{track_id}' + " " + f'{label}'
            # txt_tobe_print = f'{track_id}'
            # #
            # plot_one_box(bbox, frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)

            intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
                                                         Width=int(bbox[2] - bbox[0]),
                                                         Height=int(bbox[3] - bbox[1]),
                                                         CountUpTime=0, ObjectType="car", frameNum=self.frame_number,
                                                         ID=track_id, polygonID=int(polygon_id),
                                                         DetectionPercentage=100)

            if "car" in intruded_objects:
                intruded_objects["car"].append(intruded_object)
            else:
                intruded_objects["car"] = [intruded_object]

        detections = None
        if len(intruded_objects):
            self._last_detection_time = time.time()
            detections = self.create_structure(intruded_objects)
        else:
            if (time.time() - self._last_detection_time) >= 1:
                detections = self.send_clearance_message()

        return detections

    def create_structure(self, intruded_objects_data):
        intruded_objects_by_type = []

        for key, value in intruded_objects_data.items():
            intruded_object_by_type = ALGO_DETECTION_OBJECTS_BY_TYPE(objectCount=len(value),
                                                                     alarmSet=True,
                                                                     objectsType=key,
                                                                     algObject=value)

            intruded_objects_by_type.append(intruded_object_by_type)

        intruded_objects = ALGO_DETECTION_OBJECT(cameraId=self.camera_id,
                                                 totalObjectCount=len(intruded_objects_by_type),
                                                 videoWidth=self.video_width,
                                                 videoHeight=self.video_height,
                                                 dateTime=datetime.now(),
                                                 AlgoType=self.algo_type,
                                                 videoCounter=1,
                                                 detectionCameraConfig=self.detection_cam_config,
                                                 algoObject=intruded_objects_by_type
                                                 )

        return intruded_objects.toJSON().encode('utf-8')

    def initialize_cam_config(self):
        camera_info = CameraInfo(polygon_available=True,
                                 video_counter=1,
                                 video_width=self.video_width,
                                 video_height=self.video_height)

        return DetectionCameraConfig(cameraPolygon=self.camera_polygons, cameraInfo=camera_info)

    def initialize_zone(self):
        converted_coordinates, polygon_ids = self.get_all_polygons(self.camera_polygons)
        self.polygon_zone = TimeSpecificZone(zone_coordinates=converted_coordinates,
                                             zone_ids=polygon_ids,
                                             zone_dwelling_time_in_seconds=self._zone_dwelling_time_in_seconds)

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
            # YUv Image
            # yuv_data = np.frombuffer(buf, np.uint8).reshape(self.video_height * 3 // 2, self.video_width)
            # image_array = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_I420)

        return image_array

    def initialize_features(self, data_dict):
        try:
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
        self.initialize_zone()
        self.set_polygon_data()

    def send_clearance_message(self):
        self._last_detection_time = time.time()
        return self.create_structure({"dummy": []})
        # print("clearance sent")

    @staticmethod
    def get_all_polygons(camera_polygons):
        all_polygons = []
        polygon_ids = []

        for polygon_data in camera_polygons:
            polygon = polygon_data.Polygon
            id = polygon_data.PolygonId

            polygon_tuples = [(polygon[i], polygon[i + 1]) for i in
                              range(0, len(polygon), 2)]  # Convert to coordinate tuples
            polygon_ids.append(id)
            all_polygons.append(polygon_tuples)
        return all_polygons, polygon_ids
