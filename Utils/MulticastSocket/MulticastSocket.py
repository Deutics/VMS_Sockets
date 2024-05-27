import multiprocessing
import socket
import threading


class MulticastSocket:
    def __init__(self, multicast_address: str, multicast_port: int):
        print("new")
        self.multicast_address = multicast_address
        self.multicast_port = multicast_port
        # 2-hop restriction in network
        ttl = 2
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP,
                             socket.IP_MULTICAST_TTL,
                             ttl)

    def create_structure(self, intruded_objects):
        pass

    def send_detection(self, data):
        with multiprocessing.Lock():
            self.sock.sendto(data, (self.multicast_address, self.multicast_port))
