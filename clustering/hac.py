import sys
from typing import Callable, Union, List

import treelib

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
            tmp = self.f(n, x)
            if self.f(n, x) >= max_value:
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
                if merge_node == merge_node.parent.lchild:
                    merge_node.parent.lchild = None
                elif merge_node == merge_node.parent.rchild:
                    merge_node.parent.rchild = None
            p_parent = merge_point.parent
            parent = Node()
            if p_parent.is_left_child(merge_point):
                p_parent.lchild = parent
            elif p_parent.is_right_child(merge_point):
                p_parent.rchild = parent
            parent.lchild = merge_point
            parent.rchild = merge_node
            return parent

    def print_tree(self):
        tree = treelib.Tree()

        def traverse_tree(root: Node, parent: Union[Node, None]):
            tree.create_node("node", root, parent=parent, data=root)
            if root.lchild is not None:
                traverse_tree(root.lchild, root)
            if root.rchild is not None:
                traverse_tree(root.rchild, root)

        traverse_tree(self.dendrogram.root, None)

        tree.show(data_property="data_points")

    @staticmethod
    def lca(n1: Union[Node, None], n2: Union[Node, None]) -> Union[Node, None]:
        """
        find the lowest common ancestors of n1 and n2
        """
        if n1 is None or n2 is None:
            return None
        n1_chain = n1.ancestors
        n1_chain.append(n1)
        tmp = n2
        while tmp not in n1_chain:
            tmp = tmp.parent
            if tmp is None:
                return None
        else:
            return tmp

    @staticmethod
    def swap(s: Node, a: Node):
        s_par = s.parent
        a_par = a.parent
        if s_par.lchild == s:
            if a_par.lchild == a:
                s_par.lchild = a
                a_par.lchild = s
            elif a_par.rchild == a:
                s_par.lchild = a
                a_par.lchild = s
        elif s_par.rchild == s:
            if a_par.lchild == a:
                s_par.rchild = a
                a_par.lchild = s
            elif a_par.rchild == a:
                s_par.rchild = a
                a_par.lchild = s