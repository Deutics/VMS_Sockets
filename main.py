from Utils.MainSocketManager.MainSocketManager import MainSocketManager


if __name__ == '__main__':
    ai = MainSocketManager(server_ip="127.0.0.1",
                           server_port=7110)

    ai.get_data_from_socket()

