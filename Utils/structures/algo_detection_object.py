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
            default=lambda o: {
                "cameraLocation": o.cameraLocation,
                "cameraName": o.cameraName,
                "totalObjectCount": o.totalObjectCount,
                "videoWidth": o.videoWidth,
                "videoHeight": o.videoHeight,
                "dateTime": o.dateTime.isoformat(),  # Use isoformat() for JSON
                "AlgoType": o.AlgoType,
                "videoCounter": o.videoCounter,
                "DetectionCameraConfig": o.DetectionCameraConfig,
                "algoObject": [
                    {
                        "objectCount": ob.objectCount,
                        "alarmSet": ob.alarmSet,
                        "objectsType": ob.objectsType,
                        "algoObject": [  # Serialize each object data
                            {
                                "X": inner_ob.X,
                                "Y": inner_ob.Y,
                                "Width": inner_ob.Width,
                                "Height": inner_ob.Height,
                                "CountUpTime": inner_ob.CountUpTime,
                                "ObjectType": inner_ob.ObjectType,
                                "DetectionPercentage": inner_ob.DetectionPercentage if hasattr(inner_ob,
                                                                                               "DetectionPercentage") else None,
                                # Handle optional field
                                "frameNum": inner_ob.frameNum if hasattr(inner_ob, "frameNum") else None,
                                # Handle optional field
                                "ID": inner_ob.ID,
                                "polygonID": inner_ob.polygonID
                            } for inner_ob in ob.algoObject
                        ]
                    } for ob in o.algoObject
                ]
            },
            sort_keys=True,
            indent=4)
