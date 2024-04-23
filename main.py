
from Features.ZoneIntrusionDetector.ZoneIntrusionDetector import ZoneIntrusionDetector


# main function
if __name__ == '__main__':

    zi = ZoneIntrusionDetector(region_coordinates=[[10, 480], [240, 480], [240, 200], [10, 200]],
                               instance_id=1,
                               expected_objs=["person", "car", "bus", "truck", "motorcycle", "bicycle"],
                               zone_dwelling_time_in_seconds=0)

    # zi.process_video("drones/temp3.ts")
    # zi.process_video("udp://@224.1.1.1/1234")

