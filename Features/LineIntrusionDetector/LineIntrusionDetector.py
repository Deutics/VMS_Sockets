import json
import time
import numpy as np

from Utils.Line.Line import Line
from Features.utils import *
from datetime import datetime

# Tracker
from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker

# structs
from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT
from Utils.structures.camera_polygon import CameraPolygon
from Utils.structures.camera_info import CameraInfo
from Utils.structures.detection_cam_config import DetectionCameraConfig
from Utils.MemoryMappedFile.MemoryMappedFile import MemoryMappedFileHandler


class LineIntrusionDetector:
    def __init__(self, data_dict,
                 direction_to_check=None
                 ):

        self.mmp_file = [None] * 5

        if direction_to_check is None:
            direction_to_check = {"Left": True, "Right": True}

        self._direction_to_check = direction_to_check

        self._expected_objs = []

        self._polygon_zone = None

        self._object_tracker = ObjectTracker(use_gpu=True)

        self._line_position = None

        self.camera_id = data_dict['cameraId']
        self.video_index = data_dict['videoIndex']
        self.algo_type = data_dict['algoType']
        self.video_width = data_dict['videoWidth']
        self.video_height = data_dict['videoHeight']
        self.data_size = data_dict['dataSize']
        self.send_always = data_dict['sendAlways']

        self._first_frame = 1
        self.line_position = []
        self.frame_number = 0
        self.total_camera_polygons = -1
        self._last_detection_time = time.time()
        self.detection_cam_config = None
        self.source_frame_number = 0

        self.initialize_features(data_dict)
        # temp
        self.intrusion_count = 0

    def process_frame(self, frame, region_of_interest,  detections_from_detector):
        tracked_objects = self._object_tracker.process_frame(frame=frame,
                                                             predictions_from_detector=detections_from_detector)

        intruded_objects = None
        if len(tracked_objects):
            intruded_objects = self._check_line_intersection(frame)

        return intruded_objects

    def _draw_line(self, frame):
        """***************************************
        Functionality: Draw a line on image
        Parameters: frame
        Returns: Processed frame
        ******************************************"""
        return cv2.line(frame, self._line_position.start_position, self._line_position.end_position, (0, 0, 255), 2)

    def _check_line_intersection(self, frame):
        """ ****************************************
        Functionality: gets the positions of all the object and check if the intersects and generate notifications
        Parameters: frame
        Returns: None
        ******************************************** """

        tracks = self._object_tracker.tracker.get_tracks()  # all tracks
        intruded_objects = {}

        for i, track in enumerate(tracks):
            if track.is_missed or not len(track.get_state()[0]):
                continue

            angle_of_intersection = self._line_position.find_angle_of_intersection(track)

            # if not intruded
            if angle_of_intersection is None:
                continue

            if self._line_position.start_position[0] > self._line_position.end_position[0]:
                angle_of_intersection = abs(angle_of_intersection)
                angle_of_intersection = 180 - angle_of_intersection
                angle_of_intersection *= -1

                if self._line_position.start_position[1] < self._line_position.end_position[1]:
                    angle_of_intersection = angle_of_intersection * -1

            if angle_of_intersection > 180:
                angle_of_intersection = 180 - angle_of_intersection

            if (angle_of_intersection > 0 and self._direction_to_check["Left"]) or \
                    (angle_of_intersection < 0 and self._direction_to_check["Right"]) or \
                    track.notification_generated:

                bbox = track.get_state()[0]

                track_id = track.track_id

                label = self._object_tracker.classes[track.class_id.item()]
                # txt_tobe_print = f'{track_id}' + " " + f'{label}'
                # plot_one_box(bbox, self.frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)

                intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
                                                             Width=int(bbox[2] - bbox[0]),
                                                             Height=int(bbox[3] - bbox[1]),
                                                             CountUpTime=0, ObjectType="car", frameNum=self.frame_number,
                                                             ID=track_id, polygonID=self.line_position[0].PolygonId,
                                                             DetectionPercentage=100)

                if "car" in intruded_objects:
                    intruded_objects["car"].append(intruded_object)
                else:
                    intruded_objects["car"] = [intruded_object]

                track.notification_generated = True

        detections = None
        if len(intruded_objects):
            self._last_detection_time = time.time()
            detections = self.create_structure(intruded_objects)
        else:
            if (time.time() - self._last_detection_time) >= 1:
                detections = self.send_clearance_message()
        return detections

    def fetch_frame_buffer(self):
        if self.mmp_file[0] is None:
            mm_file_name = f"ShovalSCMMap_vid_{self.camera_id}"
            self.mmp_file[0] = MemoryMappedFileHandler(file_name=mm_file_name,
                                                       video_width=self.video_width,
                                                       video_height=self.video_height)

        buf = self.mmp_file[0].read_content()

        image_array = None
        if len(buf):
            # RGB image
            image_array = np.frombuffer(buf, dtype=np.uint8).reshape(self.video_height, self.video_width, 3)

            # YUv Image
            # yuv_data = np.frombuffer(buf, np.uint8).reshape(self.video_height * 3 // 2, self.video_width)
            # image_array = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_I420)

        return image_array

    def initialize_line_position(self):
        converted_coordinates = self.get_line_position(self.line_position)
        self._line_position = Line(start_position=converted_coordinates[0],
                                   end_position=converted_coordinates[1])

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
                                                 AlgoType=1,
                                                 videoCounter=1,
                                                 detectionCameraConfig=self.detection_cam_config,
                                                 algoObject=intruded_objects_by_type
                                                 )

        return intruded_objects.toJSON().encode('utf-8')

    def initialize_features(self, data_dict):

        try:
            for polygon_data in data_dict["CameraPolygon"]:
                camera_polygon = CameraPolygon(CamID=polygon_data["CamID"],
                                               PolygonId=polygon_data["PolygonId"],
                                               LineIntrusionDirection=polygon_data["LineIntrusionDirection"],
                                               DetectionAndAlertCount=polygon_data["DetectionAndAlertCount"],
                                               lineDirection=polygon_data["lineDirection"],
                                               MaxAllowed=polygon_data["MaxAllowed"],
                                               Polygon=polygon_data["Polygon"])

                self.line_position.append(camera_polygon)
                detection_types = json.loads(polygon_data["DetectionAndAlertCount"])
                temp_detections = []

                for detection_type in detection_types:
                    temp_detections.append(detection_type["AIDetectiontype"])
                    if detection_type["AIDetectiontype"] not in self._expected_objs:
                        self._expected_objs.append(detection_type["AIDetectiontype"])

                    if "lineDirection" in detection_type:
                        if detection_type["lineDirection"] == "Left":
                            self._direction_to_check = {"Left": True, "Right": False}
                        elif detection_type["lineDirection"] == "Right":
                            self._direction_to_check = {"Left": False, "Right": True}
                        elif detection_type["lineDirection"] == "Both":
                            self._direction_to_check = {"Left": True, "Right": True}
                        else:
                            self._direction_to_check = None

        except (TypeError, KeyError) as e:
            print(f"Error mapping polygons: {e}")

        self.detection_cam_config = self.initialize_cam_config()
        self.initialize_line_position()

    def initialize_cam_config(self):
        camera_info = CameraInfo(polygon_available=True,
                                 video_counter=1,
                                 video_width=self.video_width,
                                 video_height=self.video_height)

        return DetectionCameraConfig(cameraPolygon=self.line_position, cameraInfo=camera_info)

    def region_of_interest(self, frame):
        return frame

    @staticmethod
    def get_line_position(camera_polygons):
        line_position = None

        for line_data in camera_polygons:
            line_coordinates = line_data.Polygon

            line_tuples = [(line_coordinates[i], line_coordinates[i + 1]) for i in
                              range(0, len(line_data), 2)]  # Convert to coordinate tuples

            line_position = line_tuples
        return line_position

    def send_clearance_message(self):
        self._last_detection_time = time.time()
        return self.create_structure({"dummy": []})
