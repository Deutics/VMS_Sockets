# Tracker
from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker

# Zones
from Utils.Zone.Zone import Zone
from Utils.Zone.TimeSpecificZone import TimeSpecificZone
from Utils.MulticastSocket.MulticastSocket import MulticastSocket
from Utils.structures.algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA
from Utils.structures.algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from Utils.structures.algo_detection_object import ALGO_DETECTION_OBJECT

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

        self.multicast_socket = MulticastSocket(multicast_address="234.100.0.1",
                                                multicast_port=8088)
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
            # if self.is_current_time_within_bounds():
            #     # calling process_frame
            # frame = cv2.resize(frame, (720, 480))
            self.process_frame(frame)

            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        if self._first_frame:
            self.video_height, self.video_width, _ = frame.shape
            self._polygon_zone.create_polygon_mask(img_size=(self.video_height, self.video_width))
            self._first_frame = 0

        tracked_objects = self._object_tracker.process_frame(frame)

        frame = self._draw_polygon(frame)
        if len(tracked_objects):
            # Feature
            draw_boundary_boxes(detections=tracked_objects, img=frame)
            self._check_zone_intrusion(frame=frame, detections=tracked_objects)

        return tracked_objects

    def _draw_polygon(self, frame):

        pts = self._polygon_zone.zone_coordinates.reshape((-1, 1, 2))
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
            if not track.is_missed:
                if self._polygon_zone.if_zone_intruded(track):
                    bbox = track.get_state()[0]
                    track_id = track.track_id
                    label = self._object_tracker.classes[track.class_id.item()]

                    txt_tobe_print = f'{track_id}' + " " + f'{label}'
                    plot_one_box(bbox, frame, label=txt_tobe_print, color=(0, 0, 255), line_thickness=1)

                    # intruded_object = {"tracker_id": track.track_id,
                    #                    "label": self._object_tracker.classes[track.class_id.item()],
                    #                    "bbox": bbox}
                    # #
                    # intruded_objects.append(intruded_object)

                    intruded_object = ALGO_DETECTION_OBJECT_DATA(X=int(bbox[0]), Y=int(bbox[1]),
                                                                 Width=int(bbox[2]-bbox[0]), Height=int(bbox[3]-bbox[1]),
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

        intruded_objects = ALGO_DETECTION_OBJECT(cameraLocation=self.camera_location,
                                                 cameraName=self.camera_name,
                                                 totalObjectCount=len(intruded_objects_by_type),
                                                 videoWidth=self.video_width,
                                                 videoHeight=self.video_height,
                                                 dateTime=datetime.now(),
                                                 AlgoType=1,
                                                 videoCounter=1,
                                                 DetectionCameraConfig= "udp_multicast",
                                                 algoObject=intruded_objects_by_type
                                                 )

        self.multicast_socket.send_detection(intruded_objects.toJSON().encode('utf-8'))
