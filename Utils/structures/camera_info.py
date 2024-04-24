
class CameraInfo:
    def __init__(self, polygon_available: bool, video_counter: int, video_width: int, video_height: int):
        self.PolygonAvailable = polygon_available
        self.videoCounter = video_counter
        self.videoWidth = video_width
        self.videoHeight = video_height
