from typing import Tuple

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from gendataset.realworld_dataset import generate_realworld_dataset
from gendataset.shuffle import random_shuffle
from linkage.vector import cosine_similarity_for_binary_vector_data_point
from monitor.dendrogram_purity import DpMonitor


if __name__ == "__main__":
    # read dataset from ALOI
    # other shuffle functions are provided in gendataset/shuffle.py
    data_stream, ground_truth = generate_realworld_dataset(shuffle=random_shuffle)

    # settings of Grinch algorithm
    single_nn_search = True
    k_nn = 25

    single_elimination = True,

    capping = Tuple
    capping_height = 100

    navigable_small_world_graphs = False
    k_nsw = 50

    # the purity monitor to monitor the dendrogram purity during inserting data points
    monitor = DpMonitor(n_data_points=len(data_stream), n_workers=8, ground_truth=ground_truth)
    # grinch approach to cluster
    # other linkage functions are provided in linkage/*.py
    # customized linkage function is also allowed as long as it takes same inputs and return same outputs.
    clustering = Grinch(cosine_similarity_for_binary_vector_data_point, debug=False,
                        single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw, monitor=monitor)
    # process data stream
    for dp in data_stream:
        clustering.insert(dp)
    # wait for monitor to finish its tasks (because it's multiprocessing)
    monitor.join()
    monitor.show_purity_plot()
    clustering.dendrogram.print()
    print("dendrogram purity:", dendrogram_purity(ground_truth, clustering.dendrogram))
