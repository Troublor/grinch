import sys

from model.cluster import Cluster


def single_linkage(c1: Cluster, c2: Cluster) -> float:
    max_value = sys.float_info.min
    for p1 in c1.data_points:
        for p2 in c2.data_points:
            s = p1.pairwise_similarity(p2)
            if s > max_value:
                max_value = s
    return max_value
