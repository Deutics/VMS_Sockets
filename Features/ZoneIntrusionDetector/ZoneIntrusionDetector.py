# Tracker
from Utils.Tracker.DeepSort.ObjectTracker import ObjectTracker

# Zones
from Utils.Zone.Zone import Zone
from Utils.Zone.TimeSpecificZone import TimeSpecificZone


from Features.utils import *
from datetime import datetime


class ZoneIntrusionDetector:
    def __init__(self, region_coordinates,
                 instance_id,
                 expected_objs=None,
                 object_size=None,
                 zone_dwelling_time_in_seconds=None,
                 time_bounds=None):

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
        self._first_frame = 1

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
            if self.is_current_time_within_bounds():
                # calling process_frame
                frame = cv2.resize(frame, (720, 480))
                self.process_frame(frame)

            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        if self._first_frame:
            height, width, _ = frame.shape
            self._polygon_zone.create_polygon_mask(img_size=(height, width))
            self._first_frame = 0

        tracked_objects = self._object_tracker.process_frame(frame)

        frame = self._draw_polygon(frame)
        if len(tracked_objects):
            # Feature
            # draw_boundary_boxes(detections=tracked_objects, img=frame)
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

        for i, track in enumerate(tracks):
            if not track.is_missed:
                if self._polygon_zone.if_zone_intruded(track):
                    bbox = track.get_state()[0]
                    track_id = track.track_id
                    print(track_id, track.state)
                    plot_one_box(bbox, frame, label=str(track_id), color=(0, 0, 255), line_thickness=1)
                    # track.notification_generated = True

    def is_current_time_within_bounds(self):
        # Get current time
        current_time = datetime.now().time()

        # Converting starting and ending time strings to datetime objects
        for i, time_bound in enumerate(self._time_bounds):
            starting_time = datetime.strptime(time_bound["starting_time"], "%H:%M").time()
            ending_time = datetime.strptime(time_bound["ending_time"], "%H:%M").time()
            # Check if current time falls within the range
            if starting_time <= current_time <= ending_time:
                return True

        return False
