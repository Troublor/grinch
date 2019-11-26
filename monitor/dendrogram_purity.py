import json
from itertools import accumulate
from multiprocessing import Queue, Pool
from typing import List, Tuple
from matplotlib import pyplot

from clustering.evaluation import dendrogram_purity
from dendrogram import Tree
from model.cluster import GroundTruthCluster


class DpMonitor:
    """
    Thr monitor to monitor dendrogram purity during clustering
    """
    def __init__(self, n_data_points: int, n_workers: int, ground_truth: List[GroundTruthCluster]):
        """
        :param n_data_points: the number of data points in total
        :param n_workers: the number of processors used to calculate dendrogram purity concurrently
        :param ground_truth: clustering ground truth
        """
        self.n_data_points = n_data_points
        self.ground_truth = ground_truth
        self.pool = Pool(processes=n_workers)
        # a list: the dendrogram purity before and after graft subroutine with regard to each data point insertion
        self.dp_over_time = [[1, 1] for _ in range(n_data_points)]

    def feed(self, index: int, dendrogram: Tree, before: bool):
        """
        provide dendrogram tree to calculate purity
        :param index: the index of data point insertion
        :param dendrogram: dendrogram tree
        :param before: whether this is before graft
        """
        self.pool.apply_async(self.slave_worker, args=(index, before, self.ground_truth, dendrogram,),
                              callback=self.slave_callback)

    def join(self):
        """
        wait for tasks to finish and exit monitoring
        :return:
        """
        self.pool.close()
        self.pool.join()

    @staticmethod
    def leader_worker(dp_over_time: list, result_queue: Queue):
        index, purity = result_queue.get()
        dp_over_time[index] = purity

    @staticmethod
    def slave_worker(index: int, before: bool, ground_truth: List[GroundTruthCluster], tree: Tree):
        return index, before, dendrogram_purity(ground_truth, tree)

    def slave_callback(self, data: Tuple[int, float]):
        if data[1] is True:
            self.dp_over_time[data[0]][0] = data[2]
        else:
            # print("after inserting", data[0], "dp =", data[2])
            self.dp_over_time[data[0]][1] = data[2]

    def output_history(self, file):
        with open(file, "w") as file:
            file.write(json.dumps(self.dp_over_time))

    def show_purity_plot(self):
        x = [i for i in range(self.n_data_points)]
        y = list(map(lambda item: item[1], self.dp_over_time))
        pyplot.plot(x, y, 'b--')
        pyplot.xlabel('Data Points')  # X轴标签
        pyplot.ylabel("Dendrogram Purity")  # Y轴标签
        pyplot.legend()
        pyplot.show()

    def show_purity_change_before_and_after_graft_plot(self):
        x = [i for i in range(self.n_data_points)]
        y_inst = list(map(lambda item: item[1] - item[0], self.dp_over_time))
        y_accu = list(accumulate(y_inst, lambda a, b: a + b))
        pyplot.plot(x, y_inst, 'r--', marker="o", label='instantaneous')
        pyplot.plot(x, y_accu, 'b--', marker="o", label='accumulate')
        pyplot.xlabel('Data Points')  # X轴标签
        pyplot.ylabel("Dendrogram Purity Change")  # Y轴标签
        pyplot.legend()
        pyplot.show()

    def save_plot(self, path):
        x = [i for i in range(self.n_data_points)]
        y_inst = list(map(lambda item: item[1] - item[0], self.dp_over_time))
        y_accu = list(accumulate(y_inst, lambda a, b: a + b))
        pyplot.plot(x, y_inst, 'r--', marker="o", label='instantaneous')
        pyplot.plot(x, y_accu, 'b--', marker="o", label='accumulate')
        pyplot.xlabel('Data Points')  # X轴标签
        pyplot.ylabel("Dendrogram Purity Change")  # Y轴标签
        pyplot.legend()
        pyplot.savefig(path)
