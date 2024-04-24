# Tracker
import cv2

from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker

# Zones
from Utils.Zone.Zone import Zone
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

import threading


class ZoneIntrusionDetector:
    def __init__(self, region_coordinates,
                 instance_id,
                 camera_location,
                 camera_name,
                 expected_objs=None,
                 object_size=None,
                 zone_dwelling_time_in_seconds=None,
                 time_bounds=None,
                 ):

        if time_bounds is None:
            time_bounds = [{"starting_time": "0:0", "ending_time": "23:59"}]

        self._time_bounds = time_bounds

        self._obj_size = object_size
        self._instance_id = instance_id
        self._expected_objs = expected_objs
        self._zone_dwelling_time_in_seconds = zone_dwelling_time_in_seconds

        self._polygon_zone = TimeSpecificZone(zone_coordinates=region_coordinates,
                                              zone_dwelling_time_in_seconds=self._zone_dwelling_time_in_seconds)

        self._object_tracker = ObjectTracker(use_gpu=True,
                                             obj_size=self._obj_size,
                                             expected_objs=expected_objs,
                                             detector="Yolov8",
                                             detector_model="yolov8s.pt")

        self.multicast_socket = MulticastSocket(multicast_address="224.1.1.1",
                                                multicast_port=1234)
        self.region_min_x, self.region_min_y, self.region_max_x, self.region_max_y = 0, 0, 0, 0
        self._first_frame = 1
        self.camera_location = camera_location
        self.camera_name = camera_name
        self.video_width = 0
        self.video_height = 0

    def process_video(self, streaming_source):
        """******************************
        Functionality: read the frame of video, and send it to function process image
        Parameters: None
        Returns: None
        *********************************"""

        cap = cv2.VideoCapture(streaming_source)

        while True:
            is_frame, frame = cap.read()

            if not is_frame:
                break

            self.process_frame(frame)

            cv2.imshow("Zone Intrusion", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        if self._first_frame:
            self.video_height, self.video_width, _ = frame.shape
            # self._polygon_zone.create_polygon_mask(img_size=(self.video_height, self.video_width))
            points = self._polygon_zone.zone_coordinates
            self.region_min_x, self.region_min_y = max(0, points.min(axis=0)[0]), max(0, points.min(axis=0)[1])
            self.region_max_x, self.region_max_y = min(frame.shape[1], points.max(axis=0)[0]), points.max(axis=0)[1]
            self._first_frame = 0

        extracted_rectangle = frame[self.region_min_y:self.region_max_y,
                                    self.region_min_x:self.region_max_x]

        tracked_objects = self._object_tracker.process_frame(extracted_rectangle)

        frame = self._draw_polygon(frame)

        if len(tracked_objects):
            self._check_zone_intrusion(frame=frame, detections=tracked_objects)

        return tracked_objects

    def _draw_polygon(self, frame):

        pts = self._polygon_zone.zone_coordinates.reshape((-1, 1, 2))
        frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        return frame

    # def _check_zone_intrusion(self, frame, detections):
    #     """*********************************
    #     Functionality: Check if any of detected objects has intruded the defined zone
    #     Parameter: None
    #     Return: None
    #     ************************************"""
    #     tracks = self._object_tracker.tracker.get_tracks()
    #     intruded_objects = {}
    #
    #     for i, track in enumerate(tracks):
    #         if not track.is_missed:
    #             if self._polygon_zone.if_zone_intruded(track):
    #                 bbox = track.get_state()[0]
    #                 track_id = track.track_id
    #                 label = self._object_tracker.classes[track.class_id.item()]
    #
    #                 txt_tobe_print = f'{track_id}' + " " + f'{label}'
    #                 plot_one_box(bbox, frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)
    #
    #                 intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
    #                                                              Width=int(bbox[2]-bbox[0]), Height=int(bbox[3]-bbox[1]),
    #                                                              CountUpTime=0, ObjectType=label, frameNum=0,
    #                                                              ID=track_id, polygonID=0)
    #
    #                 if label in intruded_objects:
    #                     intruded_objects[label].append(intruded_object)
    #                 else:
    #                     intruded_objects[label] = [intruded_object]
    #
    #     if len(intruded_objects):
    #         thread = threading.Thread(target=self.create_structure, args=(intruded_objects,))
    #         thread.start()

    def _check_zone_intrusion(self, frame, detections):
        """*********************************
        Functionality: Check if any of detected objects has intruded the defined zone
        Parameter: None
        Return: None
        ************************************"""
        tracks = self._object_tracker.tracker.get_tracks()
        intruded_objects = {}

        for i, track in enumerate(tracks):
            if not track.is_missed:
                bbox = track.get_state()[0]
                if not len(bbox):
                    continue

                xmin, ymin, xmax, ymax = bbox
                track_id = track.track_id
                label = self._object_tracker.classes[track.class_id.item()]
                # points = self._polygon_zone.zone_coordinates
                # min_x, min_y = max(0, points.min(axis=0)[0]), max(0, points.min(axis=0)[1])
                bbox = [xmin + self.region_min_x, ymin + self.region_min_y,
                        xmax + self.region_min_x, ymax + self.region_min_y]

                txt_tobe_print = f'{track_id}' + " " + f'{label}'
                plot_one_box(bbox, frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)
                intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
                                                             Width=int(bbox[2] - bbox[0]),
                                                             Height=int(bbox[3] - bbox[1]),
                                                             CountUpTime=0, ObjectType=label, frameNum=0,
                                                             ID=track_id, polygonID=0)

                if label in intruded_objects:
                    intruded_objects[label].append(intruded_object)
                else:
                    intruded_objects[label] = [intruded_object]

        if len(intruded_objects):
            thread = threading.Thread(target=self.create_structure, args=(intruded_objects,))
            thread.start()

    def create_structure(self, intruded_objects_data):
        intruded_objects_by_type = []
        for key, value in intruded_objects_data.items():
            intruded_object_by_type = ALGO_DETECTION_OBJECTS_BY_TYPE(objectCount=len(value),
                                                                     alarmSet=False,
                                                                     objectsType=key,
                                                                     algObject=value)
            intruded_objects_by_type.append(intruded_object_by_type)

        camera_polygon = [CameraPolygon(camID=1, polygonId=1, detectionType="car", maxAllowed=10, polygon=[1, 2, 3])]
        camera_info = CameraInfo(polygon_available=True, video_counter=1, video_width=1920, video_height=1080)

        detection_cam_config = DetectionCameraConfig(cameraPolygon=camera_polygon, cameraInfo=camera_info)

        intruded_objects = ALGO_DETECTION_OBJECT(cameraLocation=self.camera_location,
                                                 cameraName=self.camera_name,
                                                 totalObjectCount=len(intruded_objects_by_type),
                                                 videoWidth=self.video_width,
                                                 videoHeight=self.video_height,
                                                 dateTime=datetime.now(),
                                                 AlgoType=1,
                                                 videoCounter=1,
                                                 detectionCameraConfig=detection_cam_config,
                                                 algoObject=intruded_objects_by_type
                                                 )

        self.multicast_socket.send_detection(intruded_objects.toJSON().encode('utf-8'))
