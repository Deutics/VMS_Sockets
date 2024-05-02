from datetime import datetime
from typing import List
import json
from .algo_detection_objects_by_type import ALGO_DETECTION_OBJECTS_BY_TYPE
from .detection_cam_config import DetectionCameraConfig


class ALGO_DETECTION_OBJECT:

    def __init__(self, cameraId: int,
                 totalObjectCount: int,
                 videoWidth: int,
                 videoHeight: int,
                 dateTime: datetime,
                 AlgoType: int,
                 videoCounter: int,
                 detectionCameraConfig: DetectionCameraConfig,
                 algoObject: List[ALGO_DETECTION_OBJECTS_BY_TYPE]
                 ):

        self.cameraId = cameraId
        self.totalObjectCount = totalObjectCount
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.dateTime = dateTime
        self.AlgoType = AlgoType
        self.videoCounter = videoCounter
        self.detectionCameraConfig = detectionCameraConfig
        self.algoObject = algoObject

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: {
                "cameraId": o.cameraId,
                "totalObjectCount": o.totalObjectCount,
                "videoWidth": o.videoWidth,
                "videoHeight": o.videoHeight,
                "datetime": o.dateTime.isoformat(),
                "AlgoType": o.AlgoType,
                "videoCounter": o.videoCounter,
                "detectionCameraConfig": {
                    "CameraPolygon": [
                        {
                            # "CamID": cp.CamID,
                            "PolygonId": cp.PolygonId,
                            "DetectionType": cp.DetectionType,
                            "MaxAllowed": cp.MaxAllowed,
                            "Polygon": cp.Polygon
                        } for cp in o.detectionCameraConfig.CameraPolygon
                    ],
                    "CameraInfo": {
                        "PolygonAvailable": o.detectionCameraConfig.CameraInfo.PolygonAvailable,
                        "videoCounter": o.detectionCameraConfig.CameraInfo.videoCounter,
                        "videoWidth": o.detectionCameraConfig.CameraInfo.videoWidth,
                        "videoHeight": o.detectionCameraConfig.CameraInfo.videoHeight
                    }
                },
                # "detectionCameraConfig": o.detectionCameraConfig,
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
