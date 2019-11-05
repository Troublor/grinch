import math


class DataPoint:
    def pairwise_similarity(self, other) -> float:
        pass


class TrivialDataPoint(DataPoint, float):

    def pairwise_similarity(self, other) -> float:
        return math.fabs(self - other)
