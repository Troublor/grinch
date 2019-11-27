import sys

import numpy as np

from model.cluster import Cluster
from model.data_point import TrivialDataPoint

"""
Files in this package includes some linkage functions. 

A linkage function should take two Cluster objects as input parameters and output a float value, 
indicating the similarity of two clusters. 
The larger the output value is, the more similar the two clusters are.
"""


def single_linkage(c1: Cluster, c2: Cluster) -> float:
    # the single linkage function: return the largest similarity of any two data points in two clusters
    max_value = -sys.float_info.max
    for p1 in c1.data_points:
        for p2 in c2.data_points:
            s = p1.pairwise_similarity(p2)
            if s > max_value:
                max_value = s
    return max_value


def complete_linkage(c1: Cluster, c2: Cluster) -> float:
    min_value = sys.float_info.max
    for p1 in c1.data_points:
        for p2 in c2.data_points:
            s = p1.pairwise_similarity(p2)
            if s < min_value:
                min_value = s
    return min_value


def group_average_linkage(c1: Cluster, c2: Cluster) -> float:
    count = 0
    total = 0
    for p1 in c1.data_points:
        for p2 in c2.data_points:
            total += p1.pairwise_similarity(p2)
            count += 1
    return total / count


def centroid_linkage(c1: Cluster, c2: Cluster) -> float:
    m1 = np.mean(c1.data_points, axis=0)
    m2 = np.mean(c2.data_points, axis=0)
    return np.linalg.norm(m1 - m2)
