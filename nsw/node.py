import bisect
from typing import List, Dict, Tuple

import dendrogram
from model.data_point import DataPoint
from sortedcontainers import SortedList


class Node(dendrogram.Leaf):
    def __init__(self, data_point: DataPoint):
        super().__init__(data_point)
        self._neighbors: Dict[Node, float] = {}
        # sorted in ascending order, mirror from self._neighbors,
        self._order: SortedList[Tuple[float, Node]] = SortedList([])
        # self._neighbors and self._order must be kept in consistency

    @property
    def neighbors(self) -> List:
        return list(self._neighbors.keys())

    @neighbors.setter
    def neighbors(self, nodes):
        """
        :type nodes: Dict[Node, float]
        """
        self._neighbors = nodes
        self._order = SortedList([])
        for n in nodes:
            if self not in n.neighbors:
                n._neighbors[self] = nodes[n]
                n._order.add((nodes[n], self))
                self._order.add((nodes[n], n))

    def add_neighbor(self, node, similarity: float):
        """
        :type similarity: float
        :type node Node
        """
        if node not in self._neighbors:
            self._neighbors[node] = similarity
            self._order.add((similarity, node))
            if self not in node._neighbors:
                node._neighbors[self] = similarity
                node._order.add((similarity, self))

    def similarity(self, node):
        """
        :type node Node
        """
        if node not in self._neighbors:
            return None
        else:
            return self._neighbors[node]

    def similar_neighbors(self, k: int, exclude: List = None) -> List:
        tmp = []
        if exclude is None:
            exclude = []
        i = 0
        while len(tmp) != k and i < len(self._order):
            if self._order[i][1] not in exclude:
                tmp.append(self._order[i][1])
            i += 1
        return tmp
