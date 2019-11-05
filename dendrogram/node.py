import copy
from typing import Tuple, List, Union

from model.cluster import Cluster
from model.data_point import DataPoint


class Node(Cluster):
    def __init__(self):
        super().__init__()
        # parent node
        self._par: Union[Node, None] = None
        # left child node
        self._lc: Union[Node, None] = None
        # right child node
        self._rc: Union[Node, None] = None
        # descendant cache
        self._descendants: List[Node] = []

    @property
    def _disconnected(self):
        return self._par is None

    def _left_connect(self, child):
        """
        :type child: Node
        """
        assert child._disconnected
        assert self._lc is None
        self._lc = child
        child._par = self

    def _right_connect(self, child):
        """
        :type child: Node
        """
        if not child._disconnected:
            print()
        assert child._disconnected
        assert self._rc is None
        self._rc = child
        child._par = self

    def _disconnect(self):
        if self._par is None:
            return
        if self._par._lc == self:
            self._par._lc = None
            self._par = None
        elif self._par._rc == self:
            self._par._rc = None
            self._par = None

    def _disconnect_left(self):
        if self._lc is not None:
            self._lc._disconnect()

    def _disconnect_right(self):
        if self._rc is not None:
            self._rc._disconnect()

    @property
    def parent(self):
        return self._par

    @parent.setter
    def parent(self, other):
        raise Exception("parent can not be changed")

    @property
    def lchild(self):
        return self._lc

    @lchild.setter
    def lchild(self, left_child):
        """
        :type left_child: Node
        """
        if left_child is None:
            self._disconnect_left()
            parent = self._par
            rchild = self._rc
            self._disconnect()
            self._disconnect_right()
            parent._right_connect(rchild)
        else:
            left_child._disconnect()
            self._disconnect_left()
            self._left_connect(left_child)
        self._update_cache()

    @property
    def rchild(self):
        return self._rc

    @rchild.setter
    def rchild(self, right_child):
        """
        :type right_child: Node
        """
        if right_child is None:
            self._disconnect_right()
            parent = self._par
            lchild = self._lc
            self._disconnect()
            self._disconnect_left()
            parent._right_connect(lchild)
        else:
            right_child._disconnect()
            self._disconnect_right()
            self._right_connect(right_child)
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

    @property
    def sibling(self):
        if self.parent is None:
            return None
        if self == self.parent.lchild:
            return self.parent.rchild
        if self == self.parent.rchild:
            return self.parent.lchild

    @property
    def aunt(self):
        if self.parent is None:
            return None
        return self.parent.sibling

    def replace_child(self, replace, new) -> bool:
        """
        :type replace: Node
        :type new: Node
        """
        parent = new._par
        if self._lc == replace:
            self._disconnect_left()
            new._disconnect()
            self._left_connect(new)
            if parent is not None:
                parent.lchild = None
        elif self._rc == replace:
            self._disconnect_right()
            new._disconnect()
            self._right_connect(new)
            if parent is not None:
                parent.rchild = None
        else:
            return False
        return True

    def __str__(self):
        return str(self.data_points)


class Leaf(Node):
    def __init__(self, data_point: DataPoint):
        super().__init__()
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
