from dendrogram.node import Leaf, Node
from model.data_point import DataPoint
from .hac import HAC


class OnlineHAC(HAC):

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point)
        sibling = self.nearest_neighbour(leaf)
        self.make_sib(sibling, leaf)
