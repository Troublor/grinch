import copy
import math
import sys
import time
from typing import List, Tuple

import numpy as np

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from clustering.online import OnlineHAC
from clustering.rotation import RotationHAC
from gendataset.realworld_dataset import DataProcessor
from gendataset.shuffle import *
from model.cluster import GroundTruthCluster, Cluster
from model.data_point import DataPoint, BinaryDataPoint
from monitor.dendrogram_purity import DpMonitor


def is_zero_vector(vector: List[int]) -> bool:
    for e in vector:
        if e != 0:
            return False
    else:
        return True


def data_wrapper(dataset) -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    n_cluster = 0
    for index in dataset[0]:
        if index >= n_cluster:
            n_cluster = index + 1
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


def param_print(*params):
    for param in params:
        print(param, '=', repr(eval(param)))


stdout = sys.stdout
with open("experiment/synthetic/graft_significance.txt", "w") as file:
    sys.stdout = stdout

    total_time = 0
    n_repeat = 1
    n_workers = 16
    plot_path = "experiment/graft_significance/graft.jpg"
    for i in range(n_repeat):
        gen = DataProcessor()

        single_nn_search = False
        k_nn = 25
        param_print("single_nn_search", "k_nn")

        single_elimination = False,
        param_print("single_elimination")

        capping = False
        capping_height = 100
        param_print("capping", "capping_height")

        navigable_small_world_graphs = False
        k_nsw = 50
        param_print("navigable_small_world_graphs", "k_nsw")

        output = gen.read_imgs()
        data_stream, ground_truth = data_wrapper(output)

        monitor = DpMonitor(n_data_points=len(data_stream), n_workers=n_workers, ground_truth=ground_truth)

        clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                            single_elimination=single_elimination,
                            capping=capping, capping_height=capping_height,
                            navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw, monitor=monitor)
        # clustering = OnlineHAC(cosine_similarity)
        # clustering = RotationHAC(cosine_similarity)
        count = 0
        start = time.time()
        for dp in data_stream:
            print("insert data point", count)
            clustering.insert(dp)
            count += 1
        # clustering.dendrogram.print()
        end = time.time()
        # print("rotation:", grinch.rotation_count)
        # print("graft:", grinch.graft_count)
        # print("restruct:", grinch.restruct_count)
        # print("similarity:", grinch.similarity_count)
        # print("reuse:", grinch.similarity_reused_count)
        print("clustering time:", end - start)
        total_time += end - start
        print("dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
        monitor.join()
        # monitor.show_plot()
        monitor.save_plot(plot_path)
        print("=======================================================================================================")
    print("average time:", total_time / n_repeat)
sys.stdout = stdout
