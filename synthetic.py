import math
import sys
from typing import List, Tuple

import numpy as np

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from gendataset.generate_dataset import DataGeneration
from model.cluster import GroundTruthCluster, Cluster
from model.data_point import DataPoint, BinaryDataPoint


def data_wrapper(dataset, n_cluster: int) -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    count = [0, 0, 0]
    cc = [[] for i in range(n_cluster)]
    data_stream = []
    for index, cluster in enumerate(dataset[0]):
        count[cluster] += 1
        if cluster == 0:
            dp = BinaryDataPoint(dataset[1][index], "-" + str(count[cluster]))
            cc[cluster].append(dp)
            data_stream.append(dp)
        elif cluster == 1:
            dp = BinaryDataPoint(dataset[1][index], "+" + str(count[cluster]))
            cc[cluster].append(dp)
            data_stream.append(dp)
        elif cluster == 2:
            dp = BinaryDataPoint(dataset[1][index], str(count[cluster]))
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
        return -sys.float_info.max
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


gen = DataGeneration()
output = gen.gen_random_dataset(n_cluster=2, n_point_each_cluster=20, n_dim_datapoint=100)
data_stream, ground_truth = data_wrapper(output, 3)

grinch = Grinch(cosine_similarity, debug=False)
for dp in data_stream:
    grinch.insert(dp)
grinch.dendrogram.print()
print(dendrogram_purity(ground_truth, grinch.dendrogram))
