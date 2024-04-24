
from Features.ZoneIntrusionDetector.ZoneIntrusionDetector import ZoneIntrusionDetector


# main function
if __name__ == '__main__':

    zi = ZoneIntrusionDetector(region_coordinates=[[20, 0], [800, 0], [800, 900], [20, 900]],
                               instance_id=1,
                               expected_objs=["person", "car", "bus", "truck", "motorcycle",
                                              "bicycle", "train", "boat", "suitcase"],
                               zone_dwelling_time_in_seconds=0,
                               camera_name="road",
                               camera_location="office")

    zi.process_video("udp://@234.7.7.7:7777?overrun_nonfatal=1&fifo_size=50000000")
    # zi.process_video("udp://@224.1.1.1/1234")

