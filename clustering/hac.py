import sys
from typing import Callable, Union, List, Dict

from dendrogram.node import Node, Leaf
from dendrogram.tree import Tree
from model.cluster import Cluster
from model.data_point import DataPoint


class HAC:
    def __init__(self, f: Callable[[Cluster, Cluster], float]):
        # the dendrogram tree, only store its root node
        self.dendrogram: Tree = Tree()
        # the linkage function
        self.f: Callable[[Cluster, Cluster], float] = f

        self.similarity_reused_count = 0
        self.similarity_count = 0

        self._similarity_table: Dict[Node, Dict[Node, float]] = {}

    def insert(self, data_point: DataPoint):
        pass

    def nearest_neighbour(self, x: Node) -> Union[Node, None]:
        """
        search for the nearest node(cluster)
        """
        return self.constr_nearest_neighbour(x, [])

    def constr_nearest_neighbour(self, x: Node, exclude: List[Cluster]) -> Union[Node, None]:
        # search among leaves
        if self.dendrogram.root is None:
            return None
        descendants = self.dendrogram.descendants
        max_value = -sys.float_info.max
        nearest = None
        for n in descendants:
            if n in exclude or not isinstance(n, Leaf):
                continue
            tmp = self.get_similarity(n, x)
            if tmp >= max_value:
                max_value = tmp
                nearest = n
        return nearest

    def make_sib(self, merge_point: Node, merge_node: Node) -> Node:
        """
        make_sib merge merge_point to merge_node by creating a new node,
            whose children is merge_point and merge_node and
            which is attached to the parent of merge_point
        returns the created node
        """
        if merge_point is None:
            self.dendrogram.root = merge_node
            return merge_node
        elif merge_point.parent is None:
            root = Node()
            root.lchild = merge_point
            root.rchild = merge_node
            self.dendrogram.root = root
            return root
        else:
            if merge_node.parent is not None:
                if merge_node.parent.is_left_child(merge_node):
                    merge_node.parent.lose_left_child()
                elif merge_node.parent.is_right_child(merge_node):
                    merge_node.parent.lose_right_child()
                else:
                    raise Exception("tree link inconsistent")
            p_parent = merge_point.parent
            parent = Node()
            if p_parent.is_left_child(merge_point):
                p_parent.lchild = parent
            elif p_parent.is_right_child(merge_point):
                p_parent.rchild = parent
            else:
                raise Exception("tree link inconsistent")
            parent.lchild = merge_point
            parent.rchild = merge_node
            return parent

    def get_similarity(self, n1: Node, n2: Node) -> float:
        self.similarity_count += 1
        if n1 in self._similarity_table and n2 in self._similarity_table[n1] and not n1.updated and not n2.updated:
            # print("similarity reused")
            self.similarity_reused_count += 1
            return self._similarity_table[n1][n2]
        # print("similarity update")
        sim = self.f(n1, n2)
        if n1 not in self._similarity_table:
            self._similarity_table[n1] = {n2: sim}
        else:
            self._similarity_table[n1][n2] = sim
        if n2 not in self._similarity_table:
            self._similarity_table[n2] = {n1: sim}
        else:
            self._similarity_table[n2][n1] = sim
        n1.updated = False
        n2.updated = False
        return sim

