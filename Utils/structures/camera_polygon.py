from typing import List
import json


class CameraPolygon:
    def __init__(self, polygon_data):
        self.CamID = polygon_data["CamID"],
        self.PolygonId = polygon_data["PolygonId"],
        self.LineIntrusionDirection = polygon_data["LineIntrusionDirection"],
        self.DetectionAndAlertCount = polygon_data["DetectionAndAlertCount"],
        self.lineDirection = polygon_data["lineDirection"],
        self.MaxAllowed = polygon_data["MaxAllowed"],
        self.Polygon = polygon_data["Polygon"]
        self.MaxAllowed = int(self.MaxAllowed[0])
        self.PolygonId = int(self.PolygonId[0])
        self.DetectionAndAlertCount = json.loads(self.DetectionAndAlertCount[0])

        # self.CamID = 61
        # self.PolygonId = 1
        # self.LineIntrusionDirection = []
        # self.DetectionAndAlertCount = [{}]
        # self.lineDirection = []
        # self.MaxAllowed = 10
        # self.Polygon = [1,2,4,5,6,7,8]
