import json
import mmap

import time
import socket

import cv2
import numpy as np

from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker
from Utils.Line.Line import Line
from Features.utils import *
from datetime import datetime

from Utils.MulticastSocket.MulticastSocket import MulticastSocket

# structs
from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT
from Utils.structures.camera_polygon import CameraPolygon
from Utils.structures.camera_info import CameraInfo
from Utils.structures.detection_cam_config import DetectionCameraConfig


class LineIntrusionDetector:
    def __init__(self, server_ip,
                 server_port,
                 zone_dwelling_time_in_seconds=None,
                 time_bounds=None,
                 direction_to_check=None
                 ):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((server_ip, server_port))

        except (socket.error, ConnectionRefusedError, TimeoutError) as err:
            print("Socket connection error:", err)

        if time_bounds is None:
            time_bounds = [{"starting_time": "0:0", "ending_time": "23:59"}]

        if direction_to_check is None:
            direction_to_check = {"left": True, "right": True}

        self._time_bounds = time_bounds
        self._direction_to_check = direction_to_check

        self._time_bounds = time_bounds

        self.multicast_socket = MulticastSocket(multicast_address="234.100.0.1",
                                                multicast_port=8088)

        self._expected_objs = ""
        self._zone_dwelling_time_in_seconds = zone_dwelling_time_in_seconds

        self._polygon_zone = None

        self._object_tracker = None

        self._line_position = None

        self._first_frame = 1
        self.camera_id = 0
        self.video_width = 0
        self.video_height = 0
        self.algo_type = 0
        self.camera_polygons = None
        self.total_camera_polygons = -1
        self.data_size = 0
        self.frame_number = 0
        self._last_detection_time = 0
        self.detection_cam_config = None

    def process_video(self):
        """******************************
        Functionality: read the frame of video, and send it to function process image
        Parameters: None
        Returns: None
        *********************************"""

        while True:
            frame = self.get_frame_from_socket()

            if frame is None:
                continue

            self.process_frame(frame)

            cv2.imshow("Line Intrusion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        # cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """ ********************************
        Functionality: Takes a frame send it ot tracker and gets detections
        then draw defined line and boundary boxed and then check if line intersects
        Parameters: frame(in cv2 format)
        Returns: None
        *********************************** """

        tracked_objects = self._object_tracker.process_frame(frame)
        frame = self._draw_line(frame)

        if len(tracked_objects):
            draw_boundary_boxes(detections=tracked_objects, img=frame)
            self._check_line_intersection(frame)

        return tracked_objects

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
            if track.is_missed or track.notification_generated or not len(track.get_state()[0]):
                continue

            angle_of_intersection = self._line_position.find_angle_of_intersection(track)

            # if not intruded
            if angle_of_intersection is None:
                continue

            bbox = track.get_state()[0]

            xmin, ymin, xmax, ymax = bbox
            track_id = track.track_id

            label = self._object_tracker.classes[track.class_id.item()]
            txt_tobe_print = f'{track_id}' + " " + f'{label}'
            plot_one_box(bbox, frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)

            intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
                                                         Width=int(bbox[2] - bbox[0]),
                                                         Height=int(bbox[3] - bbox[1]),
                                                         CountUpTime=0, ObjectType=label, frameNum=self.frame_number,
                                                         ID=track_id, polygonID=self.camera_polygons[0].PolygonId,
                                                         DetectionPercentage=100)

            if label in intruded_objects:
                intruded_objects[label].append(intruded_object)
            else:
                intruded_objects[label] = [intruded_object]

            track.notification_generated = True

        if len(intruded_objects):
            self._last_detection_time  = time.time()
            self.create_structure(intruded_objects)
        else:
            if time.time() - self._last_detection_time >= 1:
                self.send_clearance_message()

    def get_frame_from_socket(self):
        data, client_address = self.server_socket.recvfrom(1024)
        print(data)

        data_dict = data.decode("utf-8")

        try:
            data_dict = json.loads(data_dict)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in data_type")

        frame = None
        if int(data_dict["Opcode"]) == 10:
            print(data_dict)
            self.camera_id = data_dict['cameraId']
            self.algo_type = data_dict['algoType']
            self.video_width = data_dict['videoWidth']
            self.video_height = data_dict['videoHeight']
            self.data_size = data_dict['dataSize']

            try:
                self.camera_polygons = [
                    CameraPolygon(**polygon_data) for polygon_data in data_dict["CameraPolygon"]
                ]

            except (TypeError, KeyError) as e:
                print(f"Error mapping polygons: {e}")

            self._expected_objs = self.camera_polygons[0].DetectionType.split(",")
            self._expected_objs = [expected_object.strip() for expected_object in self._expected_objs]

            camera_info = CameraInfo(polygon_available=True,
                                     video_counter=1,
                                     video_width=self.video_width,
                                     video_height=self.video_height)

            self.detection_cam_config = DetectionCameraConfig(cameraPolygon=self.camera_polygons, cameraInfo=camera_info)
            self.initialize_ai_instance()

        elif int(data_dict["Opcode"]) == 1:
            print("opcode1")
            buffer_index = data_dict["bufferIndex"]
            self.frame_number = data_dict["frameNumber"]

            frame = self.process_buffer(buffer_index)

        return frame

    def process_buffer(self, buffer_index):

        mm_file_name = f"ShovalSCMMap{buffer_index}_vid_{0}"
        #print(mm_file_name)
        with (mmap.mmap(-1, self.video_width * self.video_height * 3, access=mmap.ACCESS_READ, tagname=mm_file_name)
              as mm):

            buf = mm.read()


        image_array = np.frombuffer(buf, dtype=np.uint8).reshape(self.video_height, self.video_width, 3)

        return image_array

    def initialize_ai_instance(self):

        converted_coordinates = self.get_all_polygons(self.camera_polygons)
        self._line_position = Line(start_position=converted_coordinates[0],
                                   end_position=converted_coordinates[1])

        self._object_tracker = ObjectTracker(use_gpu=True,
                                             obj_size=None,
                                             expected_objs=self._expected_objs,
                                             detector="Yolov8",
                                             detector_model="yolov8s.pt")

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

        self.multicast_socket.send_detection(intruded_objects.toJSON().encode('utf-8'))

    @staticmethod
    def get_all_polygons(camera_polygons):
        all_polygons = None

        for polygon_data in camera_polygons:
            polygon = polygon_data.Polygon

            polygon_tuples = [(polygon[i], polygon[i + 1]) for i in
                              range(0, len(polygon), 2)]  # Convert to coordinate tuples

            all_polygons = polygon_tuples
        return all_polygons

    def send_clearance_message(self):

        self.create_structure({"dummy": []})
