import math
from typing import List


class DataPoint:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        if self.id is not None:
            return self.id
        else:
            return str(self)

    def __eq__(self, other):
        return self.id == other.id

    def pairwise_similarity(self, other) -> float:
        pass


class TrivialDataPoint(DataPoint, float):

    def pairwise_similarity(self, other) -> float:
        if self * other > 0:
            if math.fabs(self - other) <= 1:
                return 2
        return 1


class BinaryDataPoint(DataPoint):
    def __init__(self, vector: List[int], id):
        super().__init__(id)
        self.vector = vector