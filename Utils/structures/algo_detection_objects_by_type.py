from typing import List
from .algo_detection_object_data import ALGO_DETECTION_OBJECT_DATA


class ALGO_DETECTION_OBJECTS_BY_TYPE:

    def __init__(self, objectCount: int,
                 alarmSet: bool,
                 objectsType: str,
                 algObject: List[ALGO_DETECTION_OBJECT_DATA]
                 ):
        self.objectCount = objectCount
        self.alarmSet = alarmSet
        self.objectsType = objectsType
        self.algoObject = algObject
