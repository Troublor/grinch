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
        # if is_zero_vector(vector):
        #     continue
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


gen = DataProcessor()

output_normal = gen.read_imgs()
data_stream_normal, ground_truth_normal = data_wrapper(output_normal)

output_random = gen.read_imgs(random_shuffle)
data_stream_random, ground_truth_random = data_wrapper(output_random)

output_round_robin = gen.read_imgs(random_shuffle)
data_stream_round_robin, ground_truth_round_robin = data_wrapper(output_round_robin)

output_sorted = gen.read_imgs(random_shuffle)
data_stream_sorted, ground_truth_sorted = data_wrapper(output_sorted)


def grinch_test_normal(data_stream, ground_truth):
    single_nn_search = True
    k_nn = 30
    # param_print("single_nn_search", "k_nn")

    single_elimination = False,
    # param_print("single_elimination")

    capping = False
    capping_height = 100
    # param_print("capping", "capping_height")

    navigable_small_world_graphs = False
    k_nsw = 50
    # param_print("navigable_small_world_graphs", "k_nsw")

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    print("Grinch HAC normal")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Grinch_normal: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Grinch_normal: clustering time:", end - start)
    print("Grinch_normal: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/grinch_normal.json")
    # monitor.save_plot(plot_path)


def grinch_test_random(data_stream, ground_truth):
    single_nn_search = True
    k_nn = 30
    # param_print("single_nn_search", "k_nn")

    single_elimination = False,
    # param_print("single_elimination")

    capping = False
    capping_height = 100
    # param_print("capping", "capping_height")

    navigable_small_world_graphs = False
    k_nsw = 50
    # param_print("navigable_small_world_graphs", "k_nsw")

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    print("Grinch HAC random")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Grinch_random: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Grinch_random: clustering time:", end - start)
    print("Grinch_random: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/grinch_random.json")
    # monitor.save_plot(plot_path)


def grinch_test_round_robin(data_stream, ground_truth):
    single_nn_search = True
    k_nn = 30
    # param_print("single_nn_search", "k_nn")

    single_elimination = False,
    # param_print("single_elimination")

    capping = False
    capping_height = 100
    # param_print("capping", "capping_height")

    navigable_small_world_graphs = False
    k_nsw = 50
    # param_print("navigable_small_world_graphs", "k_nsw")

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    print("Grinch HAC round robin")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Grinch_round_robin: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Grinch_round_robin: clustering time:", end - start)
    print("Grinch_round_robin: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/grinch_round_robin.json")
    # monitor.save_plot(plot_path)


def grinch_test_sorted(data_stream, ground_truth):
    single_nn_search = True
    k_nn = 30
    # param_print("single_nn_search", "k_nn")

    single_elimination = False,
    # param_print("single_elimination")

    capping = False
    capping_height = 100
    # param_print("capping", "capping_height")

    navigable_small_world_graphs = False
    k_nsw = 50
    # param_print("navigable_small_world_graphs", "k_nsw")

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = Grinch(cosine_similarity, debug=False, single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    print("Grinch HAC sorted")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Grinch_sorted: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Grinch_sorted: clustering time:", end - start)
    print("Grinch_sorted: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/grinch_sorted.json")
    # monitor.save_plot(plot_path)


def rotation_test_normal(data_stream, ground_truth):
    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = RotationHAC(cosine_similarity)
    print("Rotation HAC normal")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Rotation_normal: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Rotation_normal: clustering time:", end - start)
    print("Rotation_normal: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/rotation_normal.json")
    # monitor.save_plot(plot_path)


def rotation_test_random(data_stream, ground_truth):
    plot_path = "experiment/graft_significance/rotation.jpg"

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = RotationHAC(cosine_similarity)
    print("Rotation HAC random")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Rotation_random: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Rotation_random: clustering time:", end - start)
    print("Rotation_random: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/rotation_random.json")
    # monitor.save_plot(plot_path)


def rotation_test_round_robin(data_stream, ground_truth):
    plot_path = "experiment/graft_significance/rotation.jpg"

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = RotationHAC(cosine_similarity)
    print("Rotation HAC round robin")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Rotation_round_robin: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Rotation_round_robin: clustering time:", end - start)
    print("Rotation_round_robin: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/rotation_round_robin.json")
    # monitor.save_plot(plot_path)


def rotation_test_sorted(data_stream, ground_truth):
    plot_path = "experiment/graft_significance/rotation.jpg"

    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=1, ground_truth=ground_truth)

    clustering = RotationHAC(cosine_similarity)
    print("Rotation HAC sorted")
    count = 0
    start = time.time()
    for dp in data_stream:
        print("Rotation_sorted: insert data point", count)
        clustering.insert(dp)
        count += 1
    # clustering.dendrogram.print()
    end = time.time()
    monitor.feed(count - 1, copy.deepcopy(clustering.dendrogram), False)
    # print("rotation:", grinch.rotation_count)
    # print("graft:", grinch.graft_count)
    # print("restruct:", grinch.restruct_count)
    # print("similarity:", grinch.similarity_count)
    # print("reuse:", grinch.similarity_reused_count)
    print("Rotation_sorted: clustering time:", end - start)
    print("Rotation_sorted: dendrogram purity: ", dendrogram_purity(ground_truth, clustering.dendrogram))
    monitor.join()
    # monitor.show_plot()
    monitor.output_history("experiment/robustness/rotation_sorted.json")
    # monitor.save_plot(plot_path)


ps = [
    Process(target=grinch_test_normal, args=(data_stream_normal, ground_truth_normal,)),
    Process(target=rotation_test_normal, args=(data_stream_normal, ground_truth_normal,)),
]
for p in ps:
    p.start()
for p in ps:
    p.join()

# ps = [
#     Process(target=grinch_test_random, args=(data_stream_random, ground_truth_random,)),
#     Process(target=grinch_test_round_robin, args=(data_stream_round_robin, ground_truth_round_robin,)),
#     Process(target=grinch_test_sorted, args=(data_stream_sorted, ground_truth_sorted,)),
# ]
# for p in ps:
#     p.start()
# for p in ps:
#     p.join()
#
# print("=================================================================================================")
#
# ps = [
#     Process(target=rotation_test_random, args=(data_stream_random, ground_truth_random,)),
#     Process(target=rotation_test_round_robin, args=(data_stream_round_robin, ground_truth_round_robin,)),
#     Process(target=rotation_test_sorted, args=(data_stream_sorted, ground_truth_sorted,)),
# ]
# for p in ps:
#     p.start()
# for p in ps:
#     p.join()
