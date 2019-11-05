import copy
from typing import Tuple, List, Union

from model.cluster import Cluster
from model.data_point import DataPoint


class Node(Cluster):
    def __init__(self, parent):
        super().__init__()
        # parent node
        self.parent: Union[Node, None] = parent
        # left child node
        self._lc: Union[Node, None] = None
        # right child node
        self._rc: Union[Node, None] = None
        # descendant cache
        self._descendants: List[Node] = []

    @property
    def lchild(self):
        return self._lc

    @lchild.setter
    def lchild(self, left_child):
        if left_child is None:
            raise Exception("left child None")
        self._lc = left_child
        self._update_cache()

    @property
    def rchild(self):
        return self._rc

    @rchild.setter
    def rchild(self, right_child):
        if right_child is None:
            raise Exception("right child None")
        self._rc = right_child
        self._update_cache()

    def _update_cache(self):
        """
        update the cluster cache
        """
        # update self cluster cache and descendants cache
        if self._lc is None and self._rc is None:
            return
        if self._lc is not None:
            # update self cluster cache and descendants cache
            tmp1 = self._lc.data_points
            tmp2 = self._lc.descendants
            tmp2.append(self._lc)
            if self._rc is not None:
                tmp1 = tmp1 + self._rc.data_points
                tmp2 = tmp2 + self._rc.descendants
                tmp2.append(self._rc)
            self.data_points = tmp1
            self._descendants = tmp2
        elif self._rc is not None:
            tmp1 = self._rc.data_points
            tmp2 = self._rc.descendants
            tmp2.append(self._rc)
            if self._lc is not None:
                tmp1 = tmp1 + self._lc.data_points
                tmp2 = tmp2 + self._lc.descendants
                tmp2.append(self._lc)
            self.data_points = tmp1
            self._descendants = tmp2
        # after updating itself cluster cache, its parent should be notified
        if self.parent is not None:
            self.parent._update_cache()

    def ancestor_of(self, node) -> bool:
        """
        check whether self is ancestor of node
        """
        if node.parent == self:
            return True
        if node.parent is None:
            return False
        return self.ancestor_of(node.parent)

    @property
    def ancestors(self) -> List:
        """
        all ancestors of current node
        """
        anc = []
        tmp = self.parent
        while tmp is not None:
            anc.append(tmp)
            tmp = tmp.parent
        return anc

    @property
    def descendants(self) -> List:
        return list(self._descendants)

    def __str__(self):
        return str(self.data_points)


class Leaf(Node):
    def __init__(self, data_point: DataPoint, parent: Union[Node, None]):
        super().__init__(parent)
        self.data_points = [data_point]

    @property
    def lchild(self):
        return self._lc

    @lchild.setter
    def lchild(self, left_child):
        raise Exception("leaf node can not have child")

    @property
    def rchild(self):
        return self._rc

    @rchild.setter
    def rchild(self, right_child):
        raise Exception("leaf node can not have child")
