import sys
from typing import Callable, Union, List

from dendrogram.node import Node, Leaf
from dendrogram.tree import Tree
from model.cluster import Cluster
from model.data_point import DataPoint


class HAC:
    def __init__(self, f: Callable[[Cluster, Cluster], float]):
        # the dendrogram tree, only store its root node
        self.dendrogram: Tree = None
        # the linkage function
        self.f: Callable[[Cluster, Cluster], float] = f

    def insert(self, data_point: DataPoint):
        pass

    def nearest_neighbour(self, x: Node) -> Node:
        """
        search for the nearest node(cluster)
        """
        return self.constr_nearest_neighbour(x, [])

    def constr_nearest_neighbour(self, x: Node, exclude: List[Cluster]) -> Node:
        descendants = self.dendrogram.descendants
        descendants.append(self.dendrogram)
        max_value = sys.float_info.min
        nearest = None
        for n in descendants:
            if n in exclude:
                continue
            if self.f(n, x) >= max_value:
                nearest = n
        return nearest

    @staticmethod
    def lca(n1: Node, n2: Node) -> Union[Node, None]:
        """
        find the lowest common ancestors of n1 and n2
        """
        n1_chain = n1.ancestors.append(n1)
        tmp = n2
        while tmp not in n1_chain:
            tmp = tmp.parent
            if tmp is None:
                return None
        else:
            return tmp
