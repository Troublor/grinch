from typing import Callable, Tuple, List

import numpy as np
from random import choices

from model.cluster import GroundTruthCluster
from model.data_point import DataPoint, VectorDataPoint


def generate_synthetic_dataset(n_cluster=25, n_point_each_cluster=100, n_dim_datapoint=10000,
                               shuffle: Callable[[List[List[List[int]]]], Tuple[List[int], List[List[int]]]] = None) \
        -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    return data_wrapper(raw_synthetic_dataset(n_cluster, n_point_each_cluster, n_dim_datapoint, shuffle), n_cluster)


def raw_synthetic_dataset(n_cluster=25, n_point_each_cluster=100, n_dim_datapoint=10000,
                          shuffle: Callable[[List[List[List[int]]]], Tuple[List[int], List[List[int]]]] = None) -> \
        Tuple[List[int], List[List[int]]]:
    """
    This function returns a tuple, in which two element are two list objects.
    The second list contains all vectors which we will use as data set later.
    The first list is a list of cluster_id of the vectors in the second list,
    identifying to which cluster each vector belongs.
    """
    assert int(n_dim_datapoint / n_cluster) == float(n_dim_datapoint / n_cluster)
    span = int(n_dim_datapoint / n_cluster)
    values = [1, 0]
    probability = [0.1, 0.9]

    clusters = []

    for k in range(1, n_cluster + 1):
        pre, post = [0] * (span * (k - 1)), [0] * (n_dim_datapoint - span * k)
        cluster = []
        for _ in range(n_point_each_cluster):
            centroid = choices(values, probability, k=span)
            datapoint = pre + centroid + post
            cluster.append(datapoint)
        clusters.append(cluster)

    count = 0
    for cluster in clusters:
        count += len(cluster)
    assert n_cluster * n_point_each_cluster == count
    assert n_dim_datapoint == len(clusters[0][0])

    if shuffle is not None:
        return shuffle(clusters)
    index = []
    for i in range(len(clusters)):
        index += [i for _ in range(len(clusters[i]))]
    data_stream = []
    for cluster in clusters:
        data_stream += cluster
    return index, data_stream


# this data wrapper is need because the data structure of the output of generating dataset is different from
# what we need to cluster.
def data_wrapper(dataset, n_cluster: int) -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    count = [0 for i in range(n_cluster)]
    cc = [[] for i in range(n_cluster)]
    data_stream = []
    for index, cluster in enumerate(dataset[0]):
        vector = dataset[1][index]
        if is_zero_vector(vector):
            continue
        count[cluster] += 1
        dp = VectorDataPoint(np.array(dataset[1][index]), str(cluster) + "-" + str(count[cluster]))
        cc[cluster].append(dp)
        data_stream.append(dp)
    clusters = []
    for c in cc:
        clusters.append(GroundTruthCluster(c))
    return data_stream, clusters


def is_zero_vector(vector: List[int]) -> bool:
    for e in vector:
        if e != 0:
            return False
    else:
        return True
