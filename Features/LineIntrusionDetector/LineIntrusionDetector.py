import cv2

from Utils.Trackers.Sort.ObjectTracker import ObjectTracker
from Utils.Line.Line import Line
from Features.utils import *
import threading
from ..IntrusionChunkManager.IntrusionChunkManager import IntrusionChunkManager

import time
from datetime import datetime


class LineIntrusionDetector:
    def __init__(self, object_size=None, expected_objs=None,
                 line_position=None, direction_to_check=None, time_bounds=None, create_intrusion_chunk=False):

        if time_bounds is None:
            time_bounds = [{"starting_time": "0:0", "ending_time": "23:59"}]

        if direction_to_check is None:
            direction_to_check = {"left": True, "right": True}

        self._time_bounds = time_bounds
        self._direction_to_check = direction_to_check
        self._obj_size = object_size
        self._is_frame_captured = False

        self._line_position = Line(start_position=line_position["starting_point"],
                                   end_position=line_position["ending_point"])

        self._object_tracker = ObjectTracker(use_gpu=True, obj_size=object_size, expected_objs=expected_objs)
        self._stream_fps = None

        # new
        self._create_intrusion_chunk = create_intrusion_chunk
        if self._create_intrusion_chunk:
            self._frames_buffer_size = None
            self._complete_chunk_size = None
            self._frames_buffer = []
        # new

    # Functions
    def process_video(self, streaming_source):
        """******************************
        Functionality: read the frame of video, and send it to function process image
        Parameters: path of video
        Returns: None
        *********************************"""

        cap = cv2.VideoCapture(streaming_source)
        self._stream_fps = cap.get(cv2.CAP_PROP_FPS)

        # new
        if self._create_intrusion_chunk and self._frames_buffer_size is None:
            self._frames_buffer_size = cap.get(cv2.CAP_PROP_FPS) * 3        # 3 seconds frames
            self._complete_chunk_size = self._frames_buffer_size * 2        # total frames

            print("Frame buffer size", self._frames_buffer_size)
        # ###

        frame_count = 0
        start_time = time.time()

        while True:
            is_frame, frame = cap.read()

            if not is_frame:
                break

            frame = cv2.resize(frame, (720, 480))

            # calling process_frame
            if self.is_current_time_within_bounds():
                # new
                if self._create_intrusion_chunk:
                    self._frames_buffer.append(frame)
                    if len(self._frames_buffer) > self._frames_buffer_size:
                        self._frames_buffer.pop(0)
                ###########

                self.process_frame(frame)

            # ************* Extra
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Calculate FPS and display it
            fps = frame_count / elapsed_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # ************ Extra
            
            cv2.imshow(streaming_source, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """ ********************************
        Functionality: Takes a frame send it ot tracker and gets detections
        then draw defined line and boundary boxed and then check if line intersects
        Parameters: frame(in cv2 format)
        Returns: None
        *********************************** """
        tracked_objects = self._object_tracker.process_frame(frame)
        frame = self._draw_line(frame)

        if len(tracked_objects):
            draw_boundary_boxes(detections=tracked_objects, img=frame)
            self._check_line_intersection(frame)

        return tracked_objects

    def _draw_line(self, frame):
        """***************************************
        Functionality: Draw a line on image
        Parameters: frame
        Returns: Processed frame
        ******************************************"""
        return cv2.line(frame, self._line_position.start_position, self._line_position.end_position, (0, 0, 255), 2)

    def _check_line_intersection(self, frame):
        """ ****************************************
        Functionality: gets the positions of all the object and check if the intersects and generate notifications
        Parameters: frame
        Returns: None
        ******************************************** """

        tracks = self._object_tracker.tracker.get_tracks()  # all tracks
        for i, track in enumerate(tracks):
            # for intrusion chunk
            if (not track.video_chunk_saved) and (track.intrusion_chunk is not None):
                self._check_for_chunk_completion(track)

            angle_of_intersection = self._line_position.find_angle_of_intersection(track)

            # if not intruded
            if angle_of_intersection is None:
                continue

            # if intrusion occurred
            if track.intrusion_chunk is None and self._create_intrusion_chunk:
                track.intrusion_chunk = IntrusionChunkManager(past_frames_list=self._frames_buffer.copy(),
                                                              chunk_size=self._frames_buffer_size,
                                                              tracker_id=track.track_id + 1,
                                                              fps=self._stream_fps)

            if self._line_position.start_position[0] > self._line_position.end_position[0]:
                angle_of_intersection = abs(angle_of_intersection)
                angle_of_intersection = 180 - angle_of_intersection
                angle_of_intersection *= -1

                if self._line_position.start_position[1] < self._line_position.end_position[1]:
                    angle_of_intersection = angle_of_intersection * -1

            if angle_of_intersection > 180:
                angle_of_intersection = 180 - angle_of_intersection

            if (angle_of_intersection > 0 and self._direction_to_check["left"]) or \
                    (angle_of_intersection < 0 and self._direction_to_check["right"]):
                print("Notification generated for", track.track_id + 1)

                # # Just for debugging
                # if angle_of_intersection > 0:
                #     print("left")
                # else:
                #     print("right")

                # ##########
                # Add notification code here
                # ###########################
                track.notification_generated = True

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

    def _check_for_chunk_completion(self, track):
        if track.intrusion_chunk.current_buffer_size <= 1 or track.intrusion_chunk.generate_video:
            thread = threading.Thread(target=track.intrusion_chunk.save_video, args=(self._frames_buffer.copy(),))
            thread.start()
            track.video_chunk_saved = True
            track.intrusion_chunk = None
        else:
            track.intrusion_chunk.decrement_buffer_count()

