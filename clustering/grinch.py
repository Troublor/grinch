from dendrogram.node import Node
from .rotation import RotationHAC


class Grinch(RotationHAC):
    def graft(self, v: Node):
        pass

    def restruct(self, z: Node, r: Node):
        pass

    def make_sib(self, merge_point: Node, merge_node: Node) -> Node:
        """
        make_sib merge merge_point to merge_node by creating a new node,
            whose children is merge_point and merge_node and
            which is attached to the parent of merge_point
        returns the created node
        """
        pass
