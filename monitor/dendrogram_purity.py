from multiprocessing import Queue, Manager, Pool
from typing import Any, Optional, Callable, Iterable, Mapping, List, Tuple

from clustering.evaluation import dendrogram_purity
from dendrogram import Tree
from model.cluster import GroundTruthCluster


class DpMonitor:
    def __init__(self, n_data_points: int, n_workers: int, ground_truth: List[GroundTruthCluster]):
        self.ground_truth = ground_truth
        self.pool = Pool(processes=n_workers)
        self.dp_over_time = [0 for _ in range(n_data_points)]

    def feed(self, index: int, dendrogram: Tree):
        r = self.pool.apply_async(self.slave_worker, args=(index, self.ground_truth, dendrogram,), callback=self.slave_callback)

    def join(self):
        self.pool.close()
        self.pool.join()

    @staticmethod
    def leader_worker(dp_over_time: list, result_queue: Queue):
        index, purity = result_queue.get()
        dp_over_time[index] = purity

    @staticmethod
    def slave_worker(index: int, ground_truth: List[GroundTruthCluster], tree: Tree):
        return index, dendrogram_purity(ground_truth, tree)

    def slave_callback(self, data: Tuple[int, float]):
        print("get result", data)
        self.dp_over_time[data[0]] = data[1]
