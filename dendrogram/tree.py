from typing import Union

from .node import Node


class RootNode(Node):
    def __init__(self):
        super().__init__()

    @property
    def root(self) -> Node:
        return self.rchild

    @root.setter
    def root(self, _root):
        self.rchild = _root

    @property
    def lchild(self):
        return self._lc

    @lchild.setter
    def lchild(self, child):
        raise Exception("root node do not have left child")


Tree = RootNode
