import copy
from typing import List, Tuple

from model.data_point import DataPoint


class Cluster(object):
    def __init__(self):
        # cluster cache, used to record the set of data points in the cluster
        self._c: List[DataPoint] = []
        self._iter_index = 0

    def __contains__(self, item: DataPoint):
        for dp in self._c:
            if dp == item:
                return True
        else:
            return False

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self._c):
            tmp = self._c[self._iter_index]
            self._iter_index += 1
            return tmp
        else:
            raise StopIteration()

    @property
    def data_points(self) -> List[DataPoint]:
        """
        returns a copy of all data points in this cluster
        """
        return list(self._c)

    @data_points.setter
    def data_points(self, data_points: List[DataPoint]):
        self._c = data_points


class GroundTruthCluster(Cluster):
    def __init__(self, data_points: List[DataPoint]):
        super().__init__()
        self._c = data_points

    @property
    def data_points(self) -> List[DataPoint]:
        """
        returns a copy of all data points in this cluster
        """
        return list(self._c)

    @data_points.setter
    def data_points(self, data_points: List[DataPoint]):
        raise Exception("ground truth can not be changed")
