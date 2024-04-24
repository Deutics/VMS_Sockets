from typing import List


class CameraPolygon:
    def __init__(self, camID: int, polygonId: int, detectionType: str, maxAllowed: int, polygon: List[int]):
        self.CamID = camID
        self.PolygonId = polygonId
        self.DetectionType = detectionType
        self.MaxAllowed = maxAllowed
        self.Polygon = polygon
