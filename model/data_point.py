import math
from typing import List

import numpy as np


class DataPoint:
    """
    Data Point Class
    The clustering algorithm do clustering based on objects of this class
    To define a new type of data point, one may need to extend this base class
    and at least implement pairwise_similarity function
    """

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
        """
        Calculate the similarity of this data point and another data point
        :param other: another data point
        :type other: DataPoint
        :return: the similarity value
        """
        raise NotImplementedError()


class VectorDataPoint(DataPoint):
    def __init__(self, vector, id):
        """
        :param vector: numpy.array
        :param id: a unique string to identify the data point
        """
        super().__init__(str(id))
        self.vector = vector

    def pairwise_similarity(self, other) -> float:
        # Euclidean distance
        assert isinstance(other, VectorDataPoint)
        return np.linalg.norm(self.vector - other.vector)


class TrivialDataPoint(DataPoint):
    def __init__(self, v: int):
        super().__init__(str(v))
        self.v = v

    def pairwise_similarity(self, other) -> float:
        assert isinstance(other, TrivialDataPoint)
        return math.fabs(self.v - other.v)


class BinaryVectorDataPoint(VectorDataPoint):
    pass
