import random
import sys
from typing import List, Callable, Union

import dendrogram
from model.cluster import Cluster
from model.data_point import DataPoint
from .node import Node


class Graph(object):
    def __init__(self, k: int, f: Callable[[Cluster, Cluster], float]):
        self._nodes: List[Node] = []
        self._k = k
        self.f = f

    def add_data_point(self, data_point: DataPoint) -> Node:
        tmp = []
        node = Node(data_point)
        for n in self._nodes:
            tmp.append((n, self.f(n, node)))
        tmp.sort(key=lambda elem: elem[1], reverse=True)
        for n in tmp[:self._k]:
            node.add_neighbor(n[0], n[1])
        self._nodes.append(node)
        return node

    def nearest_neighbor(self, node: Node) -> Union[Node, None]:
        return self.constr_nearest_neighbor(node, exclude=[node])

    def constr_nearest_neighbor(self, node: dendrogram.Node, exclude: List[dendrogram.Node] = None) \
            -> Union[Node, None]:
        if exclude is None:
            exclude = [node]
        elif node not in exclude:
            exclude.append(node)
        tmp = list(self._nodes)
        for ex_node in exclude:
            if ex_node in tmp:
                tmp.remove(ex_node)
        if len(tmp) == 0:
            return None
        v = tmp[random.randint(0, len(tmp) - 1)]
        while True:
            max_value = -sys.float_info.max
            nearest = None
            for neighbor in v.similar_neighbors(len(v.neighbors), exclude):
                sim = self.get_similarity(node, neighbor)
                if sim > max_value:
                    max_value = sim
                    nearest = neighbor
            if max_value <= self.get_similarity(v, node):
                break
            v = nearest
        return v

    def get_similarity(self, n1: Union[dendrogram.Node, Node], n2: Union[dendrogram.Node, Node]) -> float:
        if n1 in self._nodes and n2 in self._nodes:
            t = n1.similarity(n2)
        else:
            t = self.f(n1, n2)
        if t is None:
            print()
        return t
