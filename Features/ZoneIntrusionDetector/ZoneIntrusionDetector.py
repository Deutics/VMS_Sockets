# Tracker
import json
import mmap

import time
import socket

import cv2
import numpy as np

from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker

from Utils.Zone.TimeSpecificZone import TimeSpecificZone
from Utils.MulticastSocket.MulticastSocket import MulticastSocket

from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT
from Utils.structures.camera_polygon import CameraPolygon
from Utils.structures.detection_cam_config import DetectionCameraConfig
from Utils.structures.camera_info import CameraInfo

from Features.utils import *
from datetime import datetime


class ZoneIntrusionDetector:
    def __init__(self, server_ip,
                 server_port,
                 zone_dwelling_time_in_seconds=None,
                 time_bounds=None,
                 ):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((server_ip, server_port))

        except (socket.error, ConnectionRefusedError, TimeoutError) as err:
            print("Socket connection error:", err)

        if time_bounds is None:
            time_bounds = [{"starting_time": "0:0", "ending_time": "23:59"}]

        self._time_bounds = time_bounds

        self.multicast_socket = MulticastSocket(multicast_address="234.100.0.1",
                                                multicast_port=8088)

        self._expected_objs = ""
        self._zone_dwelling_time_in_seconds = zone_dwelling_time_in_seconds

        self._polygon_zone = None

        self._object_tracker = None

        self.region_min_x, self.region_min_y, self.region_max_x, self.region_max_y = 0, 0, 0, 0
        self._first_frame = 1
        self.camera_id = 0
        self.video_width = 0
        self.video_height = 0
        self.algo_type = 0
        self.camera_polygons = None
        self.total_camera_polygons = -1
        self.data_size = 0
        self.frame_number = 0
        self.last_detection_time = 0
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

            # start_time = time.time()
            self.process_frame(frame)
            # end_time = time.time()
            # print((end_time-start_time)*1000)
            # cv2.imwrite("temp.jpg", frame)

            cv2.imshow("Zone Intrusion", frame)
            cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        # cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):

        if self._first_frame:
            self.total_camera_polygons = len(self.camera_polygons)
            # if self.total_camera_polygons == 1:
            points = self._polygon_zone.zone_coordinates[0]
            self.region_min_x, self.region_min_y = max(0, points.min(axis=0)[0]), max(0, points.min(axis=0)[1])
            self.region_max_x, self.region_max_y = min(frame.shape[1], points.max(axis=0)[0]), points.max(axis=0)[1]
            self._polygon_zone.create_polygon_mask((self.video_height, self.video_width))
            self._first_frame = 0

        region_of_interest = cv2.bitwise_and(frame, frame, mask=self._polygon_zone.polygon_binary_mask)

        if self.total_camera_polygons == 1:
            region_of_interest = region_of_interest[self.region_min_y:self.region_max_y,
                                                    self.region_min_x:self.region_max_x]
        # else:
        #     region_of_interest = cv2.bitwise_and(frame, frame, mask=self._polygon_zone.polygon_binary_mask)

        tracked_objects = self._object_tracker.process_frame(region_of_interest)

        frame = self._draw_polygon(frame)

        if len(tracked_objects):
            self._check_zone_intrusion(frame=frame, detections=tracked_objects)

        return tracked_objects

    # def no_detection_thread(self):
    #     last_detection = -1     # so it would activate for the first frame
    #     start_time = time.time()
    #     while True:
    #         print("thread_time ", time.time() - start_time)
    #         if last_detection == self.detection_counter:
    #             last_detection = self.detection_counter
    #             print("zero detection in last second")
    #             # generate coe
    #         time.sleep(1)   # sleep for 1 second

    def _draw_polygon(self, frame):
        for polygon in self._polygon_zone.zone_coordinates:
            pts = polygon.reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        return frame

    def _check_zone_intrusion(self, frame, detections):
        """*********************************
        Functionality: Check if any of detected objects has intruded the defined zone
        Parameter: None
        Return: None
        ************************************"""

        tracks = self._object_tracker.tracker.get_tracks()
        intruded_objects = {}

        for i, track in enumerate(tracks):
            if track.is_missed or not len(track.get_state()[0]):
                continue

            bbox = track.get_state()[0]
            # if not len(bbox):
            #     continue

            xmin, ymin, xmax, ymax = bbox
            track_id = track.track_id

            label = self._object_tracker.classes[track.class_id.item()]

            bbox = [xmin + self.region_min_x, ymin + self.region_min_y,
                    xmax + self.region_min_x, ymax + self.region_min_y]

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

        if len(intruded_objects):
            self._last_detection_time = time.time()
            self.create_structure(intruded_objects)
        else:
            if time.time() - self._last_detection_time >= 1:
                self.send_clearance_message()

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
            # print("workinh")
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

    def initialize_ai_instance(self):

        converted_coordinates = self.get_all_polygons(self.camera_polygons)
        self._polygon_zone = TimeSpecificZone(zone_coordinates=converted_coordinates,
                                              zone_dwelling_time_in_seconds=self._zone_dwelling_time_in_seconds)

        self._object_tracker = ObjectTracker(use_gpu=True,
                                             obj_size=None,
                                             expected_objs=self._expected_objs,
                                             detector="Yolov8",
                                             detector_model="yolov8s.pt")

    def process_buffer(self, buffer_index):

        mm_file_name = f"ShovalSCMMap{buffer_index}_vid_{0}"
        #print(mm_file_name)
        with (mmap.mmap(-1, self.video_width * self.video_height * 3, access=mmap.ACCESS_READ, tagname=mm_file_name)
              as mm):

            buf = mm.read()

        # width = 1920
        # height = 1080
        # num_channels = 3  # RGB image

        image_array = np.frombuffer(buf, dtype=np.uint8).reshape(self.video_height, self.video_width, 3)

        return image_array

    def send_clearance_message(self):
        # dummy_object = [ALGO_DETECTION_OBJECT_DATA(X=0, Y=0, Width=0, Height=0, CountUpTime=0, ObjectType="clear",
        #                                           frameNum=self.frame_number, ID=-1,
        #                                           polygonID=self.camera_polygons[0].PolygonId, DetectionPercentage=100)]

        self.create_structure({"dummy": []})


    @staticmethod
    def get_all_polygons(camera_polygons):

        all_polygons = []

        for polygon_data in camera_polygons:
            polygon = polygon_data.Polygon

            polygon_tuples = [(polygon[i], polygon[i + 1]) for i in
                              range(0, len(polygon), 2)]  # Convert to coordinate tuples

            all_polygons.append(polygon_tuples)
        return all_polygons




