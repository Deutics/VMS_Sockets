from typing import List


class CameraPolygon:
    def __init__(self, PolygonId: int, DetectionType: str, MaxAllowed: int, Polygon: List[int]):
        # self.CamID = CamID
        self.PolygonId = PolygonId
        self.DetectionType = DetectionType
        self.MaxAllowed = MaxAllowed
        self.Polygon = Polygon
