import json
import socket

import cv2

from Utils.ObjectDetectors.ObjectDetector import ObjectDetector
from Utils.MulticastSocket.MulticastSocket import MulticastSocket


class StreamsManager:
    def __init__(self, udp_port, udp_address, detector_model="yolov8s.pt"):

        self.object_detector = ObjectDetector(model_name=detector_model,
                                              use_gpu=True,
                                              conf_threshold=0.5,
                                              iou_threshold=0.5,)    # Object detector for all streams

        self.udp_port = udp_port
        self.udp_address = udp_address
        self.streams_features = {}      # DataType:dict -> cam_id(key) and ai_feature(value)
        self.multicast_socket = MulticastSocket(multicast_address="234.100.0.1", multicast_port=8088)
        self.server_socket = None
        self.bind_socket()

    def get_data_from_socket(self):
        while True:
            data, client_address = self.server_socket.recvfrom(1024)
            data_dict = data.decode("utf-8")

            try:
                data_dict = json.loads(data_dict)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in data_type")

            self.process_data(data_dict)

    def process_data(self, data_dict):
        # print(data_dict)
        if data_dict["Opcode"] == 10:       # for new stream and feature
            self.initialize_ai_feature(data_dict)

        elif data_dict["Opcode"] == 1:      # frame of stream
            camera_id = data_dict["cameraId"]
            if camera_id in self.streams_features:
                frame = self.streams_features[camera_id].fetch_frame_buffer()   # from memory mapped file

                frame = self.process_frame(camera_id, frame)
                # cv2.imshow(f"{self.udp_address}:{self.udp_port}", frame)
                # cv2.waitKey(1)

    def initialize_ai_feature(self, data_dict):
        camera_id = data_dict["cameraId"]

        if camera_id in self.streams_features:
            print("Camera already initialized")
            return

        if data_dict['algoType'] == 16:  # Opcode(16) -> Zone Intrusion
            self.streams_features[camera_id] = self.create_zone_intrusion_instance(data_dict)

        elif data_dict['algoType'] == 32:   # Opcode(32) -> Line Intrusion
            self.streams_features[camera_id] = self.create_line_intrusion_instance(data_dict)

        elif data_dict['algoType'] == 64:    # Opcode(3)  -> fire/smoke classification
            self.streams_features[camera_id] = self.create_fire_smoke_instance(data_dict)

    def process_frame(self, camera_id, frame):
        region_of_interest = self.streams_features[camera_id].region_of_interest(frame)       # region of interest

        # detected_objects = self.object_detector.process_frame_for_tracker(region_of_interest, [0,1])
        detected_objects = self.object_detector.process_frame_for_tracker(region_of_interest, [0, 2])

        if len(detected_objects):       # if objects in frame
            intruded_detections = self.streams_features[camera_id].process_frame(frame,
                                                                                 region_of_interest,
                                                                                 detected_objects)

            if intruded_detections is not None:
                self.multicast_socket.send_detection(intruded_detections)
                # print("send")

        return frame

    def bind_socket(self):

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.udp_address, self.udp_port))

        except (socket.error, ConnectionRefusedError, TimeoutError) as err:
            print("Socket connection error:", err)

    @staticmethod
    def create_zone_intrusion_instance(data_dict):
        from Features.ZoneIntrusionDetector.ZoneIntrusionDetector import ZoneIntrusionDetector
        return ZoneIntrusionDetector(data_dict)

    @staticmethod
    def create_line_intrusion_instance(data_dict):
        from Features.LineIntrusionDetector.LineIntrusionDetector import LineIntrusionDetector
        return LineIntrusionDetector(data_dict)

    @staticmethod
    def create_fire_smoke_instance(data_dict):
        from Features.FireAndSmokeClassifier.FireAndSmokeClassifier import FireAndSmokeClassifier
        return FireAndSmokeClassifier(data_dict)





