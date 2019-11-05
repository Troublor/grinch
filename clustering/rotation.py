import sys
from typing import List, Union

from dendrogram.node import Node, Leaf
from model.cluster import Cluster
from model.data_point import DataPoint
from .hac import HAC


class RotationHAC(HAC):

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point, parent=None)
        sibling = self.nearest_neighbour(leaf)
        new_node = self.make_sib(sibling, leaf)
        for v in self.dendrogram.descendants:
            while v.sibling is not None and v.aunt is not None and \
                    self.f(v, v.sibling) < self.f(v, v.aunt):
                self.rotate(v.sibling, v.aunt)

    @staticmethod
    def rotate(s: Node, a: Node):
        a_parent = a.parent
        s.parent.replace_child(s, a)
        a_parent.replace_child(a, s)

    def constr_nearest_neighbour(self, x: Node, exclude: List[Cluster]) -> Union[Node, None]:
        # search among leaves
        if self.dendrogram is None:
            return None
        descendants = self.dendrogram.descendants
        descendants.append(self.dendrogram)
        max_value = -sys.float_info.max
        nearest = None
        for n in descendants:
            if n in exclude or n is not Leaf:
                continue
            tmp = self.f(n, x)
            if self.f(n, x) >= max_value:
                max_value = tmp
                nearest = n
        return nearest
