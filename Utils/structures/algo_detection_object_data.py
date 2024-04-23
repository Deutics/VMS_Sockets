class ALGO_DETECTION_OBJECT_DATA:
    def __init__(self, X: int,
                 Y: int,
                 Width: int,
                 Height: int,
                 CountUpTime: int,
                 ObjectType: str,
                 frameNum: int,
                 ID: int,
                 polygonID: int,
                 DetectionPercentage=None):

        self.X = X
        self.Y = Y
        self.Width = Width
        self.Height = Height
        self.CountUpTime = CountUpTime
        self.ObjectType = ObjectType
        self.DetectionPercentage = DetectionPercentage
        self.frameNum = frameNum
        self.ID = ID
        self.polygonID = polygonID