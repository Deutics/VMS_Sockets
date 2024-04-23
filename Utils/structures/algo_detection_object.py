from datetime import datetime
from typing import List
import json
from .algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE


class ALGO_DETECTION_OBJECT:

    def __init__(self, cameraLocation: str,
                 cameraName: str,
                 totalObjectCount: int,
                 videoWidth: int,
                 videoHeight: int,
                 dateTime: datetime,
                 AlgoType: int,
                 videoCounter: int,
                 DetectionCameraConfig: str,
                 algoObject: List[ALGO_DETECTION_OBJECTS_BY_TYPE]
                 ):

        self.cameraLocation = cameraLocation
        self.cameraName = cameraName
        self.totalObjectCount = totalObjectCount
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.dateTime = dateTime
        self.AlgoType = AlgoType
        self.videoCounter = videoCounter
        self.DetectionCameraConfig = DetectionCameraConfig
        self.algoObject = algoObject

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)
