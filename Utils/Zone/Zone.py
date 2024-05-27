import numpy as np
import cv2


class Zone:
    def __init__(self, zone_coordinates=None, zone_ids=None):
        # print(zone_coordinates)
        self._zone_coordinates = np.array(zone_coordinates)
        self._zone_ids = zone_ids
        self._polygon_binary_mask = None

    def if_zone_intruded(self, track):
        # check if already in region
        if not track.notification_generated:

            object_position = track.past_positions.last_position()

            if self._polygon_binary_mask[object_position[1], object_position[0]]:
                return True

        return False

    def create_polygon_mask(self, img_size=None):
        binary_image = np.zeros(img_size, dtype=np.uint8)
        for i, zone_coordinates in enumerate(self._zone_coordinates):
            binary_image = cv2.fillPoly(binary_image, pts=[zone_coordinates], color=self._zone_ids[i])

        self._polygon_binary_mask = binary_image

    @property
    def zone_coordinates(self):
        return self._zone_coordinates

    @property
    def polygon_binary_mask(self):
        return self._polygon_binary_mask
