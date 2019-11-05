import sys
from typing import List, Union

from dendrogram.node import Node, Leaf
from dendrogram.tree import swap
from model.cluster import Cluster
from model.data_point import DataPoint
from .hac import HAC


class RotationHAC(HAC):

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point)
        sibling = self.nearest_neighbour(leaf)
        new_node = self.make_sib(sibling, leaf)
        for v in self.dendrogram.descendants:
            while v.sibling is not None and v.aunt is not None and \
                    self.f(v, v.sibling) < self.f(v, v.aunt):
                swap(v.sibling, v.aunt)
