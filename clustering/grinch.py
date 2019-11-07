import math
import sys
from typing import Union, List, Callable

import nsw
from dendrogram.node import Node, Leaf
from dendrogram.tree import lca, swap
from model.cluster import Cluster
from model.data_point import DataPoint
from nsw.graph import Graph
from .rotation import RotationHAC


class Grinch(RotationHAC):
    def __init__(self, f: Callable[[Cluster, Cluster], float], capping=False, capping_height=-1,
                 single_elimination=False, single_nn_search=False, k_nn=-1, navigable_small_world_graphs=False,
                 k_nsw=-1):
        super().__init__(f)
        self._capping = capping
        if self._capping:
            self._capping_height = capping_height

        self._single_elimination = single_elimination

        self._single_nn_search = single_nn_search
        if self._single_nn_search:
            self._k_nn = k_nn
            if self._k_nn < 1:
                raise Exception("the value of k for k-nn search is invalid")
            self._k_nn_leaves = []

        self._navigable_small_world_graphs = navigable_small_world_graphs
        if self._navigable_small_world_graphs:
            self._k_nsw = k_nsw
            if self._k_nsw <= 0:
                raise Exception("the value of k for navigable small world graph is invalid")
            self._nsw_graph = Graph(self._k_nsw, f)

    def nearest_neighbour(self, x: Node) -> Union[Node, None]:
        return self.constr_nearest_neighbour(x, exclude=[])

    def constr_nearest_neighbour(self, x: Node, exclude: List) -> Union[Node, None]:
        if self._navigable_small_world_graphs:
            return self._nsw_graph.constr_nearest_neighbor(x, exclude=exclude)
        else:
            return super().constr_nearest_neighbour(x, exclude)

    def insert(self, data_point: DataPoint):
        if self._navigable_small_world_graphs:
            x = self._nsw_graph.add_data_point(data_point)
        else:
            x = Leaf(data_point)
        sibling = self.nearest_neighbour(x)
        new_node = self.make_sib(sibling, x)
        while x.sibling is not None and x.aunt is not None and \
                self.get_similarity(x, x.sibling) < self.get_similarity(x.aunt, x.sibling):
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
            search_result = self.k_nn_search(v, k=1, exclude=v.lvs, search_range=self._k_nn_leaves)
            if isinstance(search_result, list) and len(search_result) > 0:
                l = search_result[0]
            else:
                l = None
        else:
            l = self.constr_nearest_neighbour(v, v.lvs)
        if v is None or l is None:
            print(123)
        v_prime = lca(v, l)
        st = v
        while v != v_prime and l != v_prime and v.sibling != l:
            print("graft")
            v_l = self.get_similarity(v, l)
            v_v_s = self.get_similarity(v, v.sibling)
            l_l_s = self.get_similarity(l, l.sibling)
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
                temp = self.get_similarity(z, a)
                if temp > max_value:
                    max_value = temp
                    m = a
            if self.get_similarity(z, z.sibling) < self.get_similarity(z, m):
                swap(z.sibling, m)
            z = z.parent

    def k_nn_search(self, x: Node, k: int = 1, exclude: List[Node] = None,
                    search_range: List[Node] = None) -> Union[List[Node], None]:
        # search among leaves
        if exclude is None:
            exclude = [x]
        elif x not in exclude:
            exclude.append(x)
        if search_range is None:
            if self.dendrogram.root is None:
                return None
            search_range = self.dendrogram.lvs
        tmp = []
        for n in search_range:
            if n in exclude:
                continue
            tmp.append((n, self.get_similarity(n, x)))
        try:
            tmp.sort(key=lambda elem: elem[1], reverse=True)
        except TypeError:
            print()
        output = []
        if k < 0:
            count = len(tmp)
        elif k > len(tmp):
            count = len(tmp)
        else:
            count = k
        for i in range(count):
            output.append(tmp[i][0])
        return output

    def get_similarity(self, n1: Node, n2: Node) -> float:
        if self._navigable_small_world_graphs:
            return self._nsw_graph.get_similarity(n1, n2)
        else:
            return self.f(n1, n2)
