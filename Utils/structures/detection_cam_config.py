from typing import List
from .camera_polygon import CameraPolygon
from .camera_info import CameraInfo


class DetectionCameraConfig:
    def __init__(self, cameraPolygon: List[CameraPolygon], cameraInfo: CameraInfo):
        self.CameraPolygon = cameraPolygon
        self.CameraInfo = cameraInfo
