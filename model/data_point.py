import math


class DataPoint:
    def pairwise_similarity(self, other) -> float:
        pass


class TrivialDataPoint(DataPoint, float):

    def pairwise_similarity(self, other) -> float:
        if self * other > 0:
            if math.fabs(self - other) <= 1:
                return 2
        return 1
