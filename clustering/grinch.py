import math
import sys
from typing import Union, List

from dendrogram.node import Node, Leaf
from model.cluster import Cluster
from model.data_point import DataPoint
from .rotation import RotationHAC


class Grinch(RotationHAC):

    def insert(self, data_point: DataPoint):
        x = Leaf(data_point)
        sibling = self.nearest_neighbour(x)
        new_node = self.make_sib(sibling, x)
        while x.sibling is not None and x.aunt is not None and \
                self.f(x, x.sibling) < self.f(x.aunt, x.sibling):
            self.rotate(x, x.aunt)
        p = x.parent
        while p is not None:
            p = self.graft(p)

    def graft(self, v: Node) -> Union[Node, None]:
        l = self.constr_nearest_neighbour(v, v.descendants)
        if v is None or l is None:
            print()
        v_prime = self.lca(v, l)
        st = v
        while v != v_prime and l != v_prime and v.sibling != l:
            if self.f(v, l) >= max(self.f(v, v.sibling), self.f(l, l.sibling)):
                z = v.sibling
                v = self.make_sib(v, l)
                self.restruct(z, self.lca(z, v))
                break
            if self.f(v, l) < self.f(l, l.sibling):
                l = l.parent
            if self.f(v, l) < self.f(v, v.sibling):
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
                self.swap(z.sibling, m)
            z = z.parent

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
