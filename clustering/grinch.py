import math
import sys
from typing import Union, List, Callable

from dendrogram.node import Node, Leaf
from dendrogram.tree import lca, swap
from model.cluster import Cluster
from model.data_point import DataPoint
from .rotation import RotationHAC


class Grinch(RotationHAC):
    def __init__(self, f: Callable[[Cluster, Cluster], float], capping=False, capping_height=-1,
                 single_elimination=False, single_nn_search=False, k_nn=-1, navigable_small_world_graphs=False):
        super().__init__(f)
        self._capping = capping
        self._capping_height = capping_height
        self._single_elimination = single_elimination
        self._single_nn_search = single_nn_search
        self._k_nn = k_nn
        self._k_nn_leaves = []
        self._navigable_small_world_graphs = navigable_small_world_graphs

    def insert(self, data_point: DataPoint):
        x = Leaf(data_point)
        sibling = self.nearest_neighbour(x)
        new_node = self.make_sib(sibling, x)
        while x.sibling is not None and x.aunt is not None and \
                self.f(x, x.sibling) < self.f(x.aunt, x.sibling):
            if self._capping and (self._capping_height < 0 or x.height > self._capping_height):
                break
            swap(x, x.aunt)
        p = x.parent
        if self._single_nn_search:
            self._k_nn_leaves = self.k_nn_search(x, k=self._k_nn)
        while p is not None:
            p = self.graft(p)

    def graft(self, v: Node) -> Union[Node, None]:
        if self._single_nn_search:
            l = self.k_nn_search(v, k=1, exclude=v.lvs, search_range=self._k_nn_leaves)
        else:
            l = self.constr_nearest_neighbour(v, v.lvs)
        if v is None or l is None:
            print()
        v_prime = lca(v, l)
        st = v
        while v != v_prime and l != v_prime and v.sibling != l:
            v_l = self.f(v, l)
            v_v_s = self.f(v, v.sibling)
            l_l_s = self.f(l, l.sibling)
            if v_l > max(v_v_s, l_l_s):
                z = v.sibling
                v = self.make_sib(v, l)
                self.restruct(z, lca(z, v))
                break
            if self._single_elimination and v_l < l_l_s and v_l < v_v_s:
                break
            if v_l <= l_l_s:
                l = l.parent
            if v_l <= v_v_s:
                v = v.parent
            if v.ancestor_of(l) or l.ancestor_of(v):
                break
        if v == st:
            return v_prime
        else:
            return v

    def restruct(self, z: Node, r: Node):
        while z != r:
            a_s = []
            for n in r.ancestors:
                if n.sibling is not None:
                    a_s.append(n.sibling)
            if len(a_s) == 0:
                return
            max_value = -sys.float_info.max
            m = None
            for a in a_s:
                temp = self.f(z, a)
                if temp > max_value:
                    max_value = temp
                    m = a
            if self.f(z, z.sibling) < self.f(z, m):
                swap(z.sibling, m)
            z = z.parent

    def k_nn_search(self, x: Node, k: int = 1, exclude: List[Node] = None,
                    search_range: List[Node] = None) -> Union[List[Node], None, Node]:
        # search among leaves
        if exclude is None:
            exclude = []
        if search_range is None:
            if self.dendrogram.root is None:
                return None
            search_range = self.dendrogram.lvs
        tmp = []
        for n in search_range:
            if n in exclude:
                continue
            tmp.append((n, self.f(n, x)))
        tmp.sort(key=lambda elem: elem[1], reverse=True)
        output = []
        if k < 0:
            count = len(tmp)
        elif k > len(tmp):
            count = len(tmp)
        else:
            count = k
        for i in search_range(count):
            output.append(tmp[i][0])
        if len(output) == 1:
            return output[0]
        else:
            return output
