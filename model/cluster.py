import copy
from typing import List, Tuple

from model.data_point import DataPoint


class Cluster:
    def __init__(self):
        # cluster cache, used to record the set of data points in the cluster
        self._c: List[DataPoint] = []

    def __contains__(self, item: DataPoint):
        for dp in self._c:
            if dp == item:
                return True
        else:
            return False

    @property
    def data_points(self) -> List[DataPoint]:
        """
        returns a copy of all data points in this cluster
        """
        return list(self._c)

    @data_points.setter
    def data_points(self, data_points: List[DataPoint]):
        self._c = data_points
