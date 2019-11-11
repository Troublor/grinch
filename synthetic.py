import math
import sys
import time
from typing import List, Tuple

import numpy as np

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from clustering.online import OnlineHAC
from clustering.rotation import RotationHAC
from gendataset.generate_dataset import DataGeneration
from gendataset.shuffle import *
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
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_similarity(c1: Cluster, c2: Cluster) -> float:
    v1 = list(map(lambda elem: elem.vector, c1.data_points))
    v2 = list(map(lambda elem: elem.vector, c2.data_points))
    return vector_cosine(np.sum(v1, axis=0), np.sum(v2, axis=0))


gen = DataGeneration(shuffle=sorted_shuffle)
n_cluster = 2
n_point_each_cluster = 25
n_dim_datapoint = 10000
output = gen.gen_random_dataset(n_cluster=n_cluster, n_point_each_cluster=n_point_each_cluster,
                                n_dim_datapoint=n_dim_datapoint)
data_stream, ground_truth = data_wrapper(output, n_cluster)

# clustering = Grinch(cosine_similarity, debug=False, single_nn_search=False, k_nn=25, single_elimination=False,
#                 capping=False, capping_height=100)
# clustering = OnlineHAC(cosine_similarity)
clustering = RotationHAC(cosine_similarity)
count = 0
start = time.time()
for dp in data_stream:
    count += 1
    # print("insert data point", count)
    clustering.insert(dp)
clustering.dendrogram.print()
end = time.time()
# print("rotation:", grinch.rotation_count)
# print("graft:", grinch.graft_count)
# print("restruct:", grinch.restruct_count)
# print("similarity:", grinch.similarity_count)
# print("reuse:", grinch.similarity_reused_count)
print("time:", end - start)
print("dp:", dendrogram_purity(ground_truth, clustering.dendrogram))
