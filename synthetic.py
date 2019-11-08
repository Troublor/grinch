import math
import sys
from typing import List, Tuple

import numpy as np

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from gendataset.generate_dataset import DataGeneration
from model.cluster import GroundTruthCluster, Cluster
from model.data_point import DataPoint, BinaryDataPoint


def is_zero_vector(vector: List[int]) -> bool:
    for e in vector:
        if e != 0:
            return False
    else:
        return True


def data_wrapper(dataset, n_cluster: int) -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    count = [0 for i in range(n_cluster)]
    cc = [[] for i in range(n_cluster)]
    data_stream = []
    for index, cluster in enumerate(dataset[0]):
        vector = dataset[1][index]
        if is_zero_vector(vector):
            continue
        count[cluster] += 1
        dp = BinaryDataPoint(dataset[1][index], str(cluster) + "-" + str(count[cluster]))
        cc[cluster].append(dp)
        data_stream.append(dp)
    clusters = []
    for c in cc:
        clusters.append(GroundTruthCluster(c))
    return data_stream, clusters


def vector_cosine(v1: List[int], v2: List[int]):
    assert len(v1) == len(v2)
    ab = 0
    a2 = 0
    b2 = 0
    for i in range(len(v1)):
        ab += v1[i] * v2[i]
        a2 += v1[i] ** 2
        b2 += v2[i] ** 2
    d = (math.sqrt(a2) * math.sqrt(b2))
    if d == 0:
        return -sys.float_info.max + 1
    return ab / d


def cosine_similarity(c1: Cluster, c2: Cluster) -> float:
    total1 = np.array(c1.data_points[0].vector)
    for dp in c1.data_points[1:]:
        assert isinstance(dp, BinaryDataPoint)
        total1 += np.array(dp.vector)
    total2 = np.array(c2.data_points[0].vector)
    for dp in c2.data_points[1:]:
        assert isinstance(dp, BinaryDataPoint)
        total2 += np.array(dp.vector)
    return vector_cosine(total1, total2)
    # return vector_cosine(c1.cache.tolist(), c2.cache.tolist())


gen = DataGeneration()
n_cluster = 4
n_point_each_cluster = 50
n_dim_datapoint = 2500
output = gen.gen_random_dataset(n_cluster=n_cluster, n_point_each_cluster=n_point_each_cluster, n_dim_datapoint=n_dim_datapoint)
data_stream, ground_truth = data_wrapper(output, n_cluster)

grinch = Grinch(cosine_similarity, debug=False, navigable_small_world_graphs=False, k_nsw=30)
count = 0
for dp in data_stream:
    count += 1
    print("insert data point", count)
    grinch.insert(dp)
grinch.dendrogram.print()
print(dendrogram_purity(ground_truth, grinch.dendrogram))
