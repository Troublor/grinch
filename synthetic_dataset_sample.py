from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from gendataset.synthetic_dataset import generate_synthetic_dataset
from gendataset.shuffle import random_shuffle
from linkage.vector import cosine_similarity_for_binary_vector_data_point


if __name__ == "__main__":
    # the setting of generating synthetic dataset
    n_cluster = 4
    n_point_each_cluster = 25
    n_dim_datapoint = 2500
    # generate synthetic dataset
    # other shuffle functions are provided in gendataset/shuffle.py
    data_stream, ground_truth = generate_synthetic_dataset(n_cluster=n_cluster, n_point_each_cluster=n_point_each_cluster,
                                   n_dim_datapoint=n_dim_datapoint, shuffle=random_shuffle)

    # settings of Grinch algorithm
    single_nn_search = False
    k_nn = 25

    single_elimination = False,

    capping = False
    capping_height = 100

    navigable_small_world_graphs = False
    k_nsw = 50
    # grinch approach to cluster
    # other linkage functions are provided in linkage/*.py
    # customized linkage function is also allowed as long as it takes same inputs and return same outputs.
    clustering = Grinch(cosine_similarity_for_binary_vector_data_point, debug=False,
                        single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    # process data stream
    for dp in data_stream:
        clustering.insert(dp)
    clustering.dendrogram.print()
    print("dendrogram purity:", dendrogram_purity(ground_truth, clustering.dendrogram))
