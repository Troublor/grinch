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
        # node height from deepest leaf
        self._height = 1

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
    def height(self):
        return self._height

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
            parent = self._par
            rchild = self._rc
            self._disconnect_left()
            self._disconnect_right()
            if parent.is_right_child(self):
                self._disconnect()
                parent._right_connect(rchild)
            elif parent.is_left_child(self):
                self._disconnect()
                parent._left_connect(rchild)
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
            parent = self._par
            lchild = self._lc
            self._disconnect_right()
            self._disconnect_left()
            if parent.is_right_child(self):
                self._disconnect()
                parent._right_connect(lchild)
            elif parent.is_left_child(self):
                self._disconnect()
                parent._left_connect(lchild)
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
        # update the height of the node
        l_height = 0
        r_height = 0
        if self._lc is not None:
            l_height = self._lc.height
        if self._rc is not None:
            r_height = self._rc.height
        self._height = max(l_height, r_height) + 1
        # after updating itself cluster cache, its parent should be notified
        if self.parent is not None:
            self.parent._update_cache()

    def lose_left_child(self):
        """
        :return: :type Node
        """
        lchild = self.lchild
        self.lchild = None
        return lchild

    def lose_right_child(self):
        """
        :return: :type Node
        """
        rchild = self.rchild
        self.rchild = None
        return rchild

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
    def lvs(self):
        tmp = []
        for n in self._descendants:
            if isinstance(n, Leaf):
                tmp.append(n)
        return tmp

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

    def is_left_child(self, child):
        return self._lc == child

    def is_right_child(self, child):
        return self._rc == child

    def __str__(self):
        return list(map(lambda item: str(item.id), self.data_points))

    def sanity_check(self):
        assert len(self.data_points) > 0
        if not isinstance(self, Leaf):
            assert self.lchild is not None
            assert self.lchild.parent == self
            self.lchild.sanity_check()
            assert self.rchild is not None
            assert self.rchild.parent == self
            self.rchild.sanity_check()
        if self.parent is not None:
            assert self.parent.is_right_child(self) or self.parent.is_left_child(self)

    @property
    def string(self):
        return list(map(lambda item: str(item.id), self.data_points))


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

    @property
    def data_point(self) -> DataPoint:
        return self.data_points[0]
