from dendrogram.node import Leaf, Node
from model.data_point import DataPoint
from .hac import HAC


class OnlineHAC(HAC):

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point, parent=None)
        sibling = self.nearest_neighbour(leaf)
        if sibling is None:
            self.dendrogram = leaf
        elif sibling.parent is None:
            root = Node(parent=None)
            root.lchild = sibling
            root.rchild = leaf
            self.dendrogram = root
        else:
            parent = Node(parent=sibling.parent)
            parent.lchild = sibling
            parent.rchild = leaf
