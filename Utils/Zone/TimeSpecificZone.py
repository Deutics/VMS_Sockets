from Utils.Zone.Zone import Zone
import time


class TimeSpecificZone(Zone):
    def __init__(self, zone_coordinates=None, zone_dwelling_time_in_seconds=None):
        Zone.__init__(self, zone_coordinates=zone_coordinates)
        self._zone_dwelling_time_in_seconds = zone_dwelling_time_in_seconds

    def if_zone_intruded(self, track):
        # check if already in region
        if not track.notification_generated:
            object_position = track.past_positions.last_position()

            if self._polygon_binary_mask[object_position[1], object_position[0]]:
                if track.time_detected is None:
                    track.time_detected = time.time()

                elif time.time() - track.time_detected > self._zone_dwelling_time_in_seconds:
                    return True

        return False
