import copy
import math
import sys
import time
from multiprocessing import Process
from typing import List, Tuple

import numpy as np

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from clustering.online import OnlineHAC
from clustering.rotation import RotationHAC
from gendataset.generate_dataset import DataGeneration
from gendataset.realworld_dataset import DataProcessor
from gendataset.shuffle import *
from model.cluster import GroundTruthCluster, Cluster
from model.data_point import DataPoint, BinaryVectorDataPoint
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
        # if is_zero_vector(vector):
        #     continue
        count[cluster] += 1
        dp = BinaryVectorDataPoint(np.array(dataset[1][index]), str(cluster) + "-" + str(count[cluster]))
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


gen = DataGeneration(shuffle=random_shuffle)
n_cluster = 20
n_point_each_cluster = 25
n_dim_datapoint = 2500
param_print("n_cluster", "n_point_each_cluster", "n_dim_datapoint")
output = gen.gen_random_dataset(n_cluster=n_cluster, n_point_each_cluster=n_point_each_cluster,
                                n_dim_datapoint=n_dim_datapoint)
data_stream, ground_truth = data_wrapper(output)


def appro_none(data_stream, ground_truth):
    with open("experiment/approximation/none.txt", "w") as file:
        total_time = 0
        n = 5
        for i in range(n):
            single_nn_search = False
            k_nn = 25

            single_elimination = False,

            capping = False
            capping_height = 100

            navigable_small_world_graphs = False
            k_nsw = 50

            clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                                single_elimination=single_elimination,
                                capping=capping, capping_height=capping_height,
                                navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
            count = 0
            start = time.time()
            for dp in data_stream:
                # print("insert data point", count)
                clustering.insert(dp)
                # cp = copy.deepcopy(clustering.dendrogram)
                # monitor.feed(count, cp)
                count += 1
            # clustering.dendrogram.print()
            end = time.time()
            # print("rotation:", grinch.rotation_count)
            # print("graft:", grinch.graft_count)
            # print("restruct:", grinch.restruct_count)
            # print("similarity:", grinch.similarity_count)
            # print("reuse:", grinch.similarity_reused_count)
            print("appro_none clustering time:", end - start)
            total_time += end - start
            purity = dendrogram_purity(ground_truth, clustering.dendrogram)
            print("appro_none dendrogram purity: ", purity)
            file.write("{}\n{}\n==========================================\n".format(end - start, purity))
            print("===================================================================================================")
        print("appro_none average time:", total_time / n)
        file.write("average time: {}".format(total_time / n))


def appro_single_nn_search(data_stream, ground_truth):
    with open("experiment/approximation/single_nn_search.txt", "w") as file:
        total_time = 0
        n = 5
        for i in range(n):
            single_nn_search = True
            k_nn = 25

            single_elimination = False,

            capping = False
            capping_height = 100

            navigable_small_world_graphs = False
            k_nsw = 50

            clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                                single_elimination=single_elimination,
                                capping=capping, capping_height=capping_height,
                                navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
            count = 0
            start = time.time()
            for dp in data_stream:
                # print("insert data point", count)
                clustering.insert(dp)
                # cp = copy.deepcopy(clustering.dendrogram)
                # monitor.feed(count, cp)
                count += 1
            # clustering.dendrogram.print()
            end = time.time()
            # print("rotation:", grinch.rotation_count)
            # print("graft:", grinch.graft_count)
            # print("restruct:", grinch.restruct_count)
            # print("similarity:", grinch.similarity_count)
            # print("reuse:", grinch.similarity_reused_count)
            print("single_nn_search clustering time:", end - start)
            total_time += end - start
            purity = dendrogram_purity(ground_truth, clustering.dendrogram)
            print("single_nn_search dendrogram purity: ", purity)
            file.write("{}\n{}\n==========================================\n".format(end - start, purity))
            print("===================================================================================================")
        print("single_nn_search average time:", total_time / n)
        file.write("average time: {}".format(total_time / n))


def appro_single_elimination(data_stream, ground_truth):
    with open("experiment/approximation/single_elimination.txt", "w") as file:
        total_time = 0
        n = 5
        for i in range(n):
            single_nn_search = True
            k_nn = 25

            single_elimination = True,

            capping = False
            capping_height = 100

            navigable_small_world_graphs = False
            k_nsw = 50

            clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                                single_elimination=single_elimination,
                                capping=capping, capping_height=capping_height,
                                navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
            count = 0
            start = time.time()
            for dp in data_stream:
                # print("insert data point", count)
                clustering.insert(dp)
                # cp = copy.deepcopy(clustering.dendrogram)
                # monitor.feed(count, cp)
                count += 1
            # clustering.dendrogram.print()
            end = time.time()
            # print("rotation:", grinch.rotation_count)
            # print("graft:", grinch.graft_count)
            # print("restruct:", grinch.restruct_count)
            # print("similarity:", grinch.similarity_count)
            # print("reuse:", grinch.similarity_reused_count)
            print("single_elimination clustering time:", end - start)
            total_time += end - start
            purity = dendrogram_purity(ground_truth, clustering.dendrogram)
            print("single_elimination dendrogram purity: ", purity)
            file.write("{}\n{}\n==========================================\n".format(end - start, purity))
            print("===================================================================================================")
        print("single_elimination average time:", total_time / n)
        file.write("average time: {}".format(total_time / n))


def appro_capping(data_stream, ground_truth):
    with open("experiment/approximation/capping.txt", "w") as file:
        total_time = 0
        n = 5
        for i in range(n):
            single_nn_search = True
            k_nn = 25

            single_elimination = True,

            capping = True
            capping_height = 100

            navigable_small_world_graphs = False
            k_nsw = 50

            clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                                single_elimination=single_elimination,
                                capping=capping, capping_height=capping_height,
                                navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
            count = 0
            start = time.time()
            for dp in data_stream:
                # print("insert data point", count)
                clustering.insert(dp)
                # cp = copy.deepcopy(clustering.dendrogram)
                # monitor.feed(count, cp)
                count += 1
            # clustering.dendrogram.print()
            end = time.time()
            # print("rotation:", grinch.rotation_count)
            # print("graft:", grinch.graft_count)
            # print("restruct:", grinch.restruct_count)
            # print("similarity:", grinch.similarity_count)
            # print("reuse:", grinch.similarity_reused_count)
            print("capping clustering time:", end - start)
            total_time += end - start
            purity = dendrogram_purity(ground_truth, clustering.dendrogram)
            print("capping dendrogram purity: ", purity)
            file.write("{}\n{}\n==========================================\n".format(end - start, purity))
            print("===================================================================================================")
        print("capping average time:", total_time / n)
        file.write("average time: {}".format(total_time / n))


def appro_navigable_small_world_graphs(data_stream, ground_truth):
    with open("experiment/approximation/navigable_small_world_graphs.txt", "w") as file:
        total_time = 0
        n = 5
        for i in range(n):
            single_nn_search = True
            k_nn = 25

            single_elimination = True,

            capping = True
            capping_height = 100

            navigable_small_world_graphs = True
            k_nsw = 50

            clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                                single_elimination=single_elimination,
                                capping=capping, capping_height=capping_height,
                                navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
            count = 0
            start = time.time()
            for dp in data_stream:
                # print("insert data point", count)
                clustering.insert(dp)
                # cp = copy.deepcopy(clustering.dendrogram)
                # monitor.feed(count, cp)
                count += 1
            # clustering.dendrogram.print()
            end = time.time()
            # print("rotation:", grinch.rotation_count)
            # print("graft:", grinch.graft_count)
            # print("restruct:", grinch.restruct_count)
            # print("similarity:", grinch.similarity_count)
            # print("reuse:", grinch.similarity_reused_count)
            print("navigable_small_world_graphs clustering time:", end - start)
            total_time += end - start
            purity = dendrogram_purity(ground_truth, clustering.dendrogram)
            print("navigable_small_world_graphs dendrogram purity: ", purity)
            file.write("{}\n{}\n==========================================\n".format(end - start, purity))
            print("===================================================================================================")
        print("navigable_small_world_graphs average time:", total_time / n)
        file.write("average time: {}".format(total_time / n))


ps = [
    Process(target=appro_none, args=(data_stream, ground_truth,)),
    Process(target=appro_single_nn_search, args=(data_stream, ground_truth,)),
    Process(target=appro_single_elimination, args=(data_stream, ground_truth,)),
    Process(target=appro_capping, args=(data_stream, ground_truth,)),
    Process(target=appro_navigable_small_world_graphs, args=(data_stream, ground_truth,)),
]

for p in ps:
    p.start()
for p in ps:
    p.join()
