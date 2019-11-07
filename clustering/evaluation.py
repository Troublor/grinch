from typing import List, Set, Tuple

from clustering import hac
from dendrogram.node import Leaf
from dendrogram.tree import Tree, lca
from model.cluster import GroundTruthCluster
from model.data_point import DataPoint


def dendrogram_purity(ground_truth: List[GroundTruthCluster], dendrogram: Tree) -> float:
    p_star_set = p_star(ground_truth, dendrogram)
    total = 0
    for (dp1, dp2) in p_star_set:
        for cluster in ground_truth:
            if dp1.data_point in cluster:
                total += pur(lca(dp1, dp2).lvs, cluster)
    return total / len(p_star_set)


def p_star(ground_truth: List[GroundTruthCluster], dendrogram: Tree) -> Set[Tuple[Leaf, Leaf]]:
    out = set()
    for cluster in ground_truth:
        data_points = cluster.data_points
        for i in range(len(data_points)):
            dp1 = data_points[i]
            for j in range(len(data_points)):
                if i == j:
                    continue
                dp2 = data_points[j]
                leaf1, leaf2 = None, None
                for leaf in dendrogram.lvs:
                    if leaf.data_point == dp1:
                        leaf1 = leaf
                    elif leaf.data_point == dp2:
                        leaf2 = leaf
                if leaf1 is None or leaf2 is None:
                    raise Exception("part of dendrogram tree is mising")
                out.add((leaf1, leaf2))
    return out


def pur(leaves: List[Leaf], cluster: GroundTruthCluster) -> float:
    n = 0
    for leaf in leaves:
        if leaf.data_point in cluster:
            n += 1
    return n / len(leaves)
