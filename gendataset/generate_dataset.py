from functools import reduce
from typing import Callable, Tuple, List

import numpy as np
from numpy import random
from random import choices


class DataGeneration(object):
    def __init__(self, shuffle: Callable[[List[List[List[int]]]], Tuple[List[int], List[List[int]]]] = None):
        self.shuffle = shuffle

    def gen_random_dataset(self, n_cluster=25, n_point_each_cluster=100, n_dim_datapoint=10000) -> \
            Tuple[List[int], List[List[int]]]:
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

        print("n_cluster = {}\nn_dim_datapoint = {}\nn_point_each_cluster = {}".format(n_cluster, n_dim_datapoint,
                                                                                       n_point_each_cluster))
        print("size of synthesized dataset = {} * {}".format(count, len(clusters[0][0])))

        if self.shuffle is not None:
            return self.shuffle(clusters)
        index = []
        for i in range(len(clusters)):
            index += [i for _ in range(len(clusters[i]))]
        data_stream = []
        for cluster in clusters:
            data_stream += cluster
        return index, data_stream
