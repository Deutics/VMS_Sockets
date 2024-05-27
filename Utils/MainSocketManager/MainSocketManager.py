import socket
import json
# import multiprocessing

from torch import multiprocessing
from Utils.StreamsManager.StreamsManager import StreamsManager


class MainSocketManager:
    def __init__(self, server_ip,
                 server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_socket = None
        self.bind_socket()

    def bind_socket(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.server_ip, self.server_port))
            print("connected to ", self.server_ip, self.server_port)
        except (socket.error, ConnectionRefusedError, TimeoutError) as err:
            print("Socket connection error:", err)

    def get_data_from_socket(self):
        while True:
            data, client_address = self.server_socket.recvfrom(1024)
            data_dict = data.decode("utf-8")
            # print(data_dict)

            try:
                data_dict = json.loads(data_dict)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in data_type")

            self.process_data(data_dict)

    def process_data(self, data_dict):
        if int(data_dict["Opcode"]) == 14:  # for creating detector instances
            self.create_detectors_pool(data_dict)

    def create_detectors_pool(self, data_dict):
        number_of_detectors = data_dict["NumberOfDetectors"]
        detector_ids = [(i + 1) for i in range(number_of_detectors)]
        # detector_ids = [1,2]

        for i, id in enumerate(detector_ids):
            process = multiprocessing.Process(target=self.start_detector, args=(id,))
            process.start()

        # pool = multiprocessing.Pool(processes=number_of_detectors)
        # pool.map_async(self.start_detector, detector_ids)
        # with multiprocessing.Pool(processes=number_of_detectors) as pool:
        #     pool.map_async(self.start_detector, detector_ids)

    @staticmethod
    def start_detector(detector_id):
        stream_manager = StreamsManager(udp_address="127.0.0.1",
                                        udp_port=7110 + detector_id)
        stream_manager.get_data_from_socket()
