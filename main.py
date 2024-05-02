
from Features.ZoneIntrusionDetector.ZoneIntrusionDetector import ZoneIntrusionDetector


# main function
if __name__ == '__main__':

    zi = ZoneIntrusionDetector(zone_dwelling_time_in_seconds=0,
                               server_ip="127.0.0.1",
                               server_port=7111)

    zi.process_video()
    # zi.process_video("udp://@234.7.7.7:7777?")
