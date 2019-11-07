import numpy as np
from numpy import random
from random import choices


class DataGeneration(object):
    def __init__(self):
        pass

    def gen_random_dataset(self, n_cluster=25, n_point_each_cluster=100, n_dim_datapoint=10000):
        # only generate binary values, generating value 1 with probability 0.1
        values = [1, 0]
        probability = [0.1, 0.9]

        gen_dataset = []
        index = []
        for k in range(0, n_cluster):
            pre, post = [0] * n_point_each_cluster * k, [0] * (n_dim_datapoint - n_point_each_cluster * (k + 1))
            for _ in range(n_point_each_cluster):
                centroid = choices(values, probability, k=n_point_each_cluster)
                datapoint = pre + centroid + post
                gen_dataset.append(datapoint)
            index.extend([k] * n_point_each_cluster)

        assert n_cluster * n_point_each_cluster == len(gen_dataset)
        assert n_dim_datapoint == len(gen_dataset[0])

        print("n_cluster = {}\nn_dim_datapoint = {}\nn_point_each_cluster = {}".format(n_cluster, n_dim_datapoint, n_point_each_cluster))
        print("size of synthesized dataset = {} * {}".format(len(gen_dataset), len(gen_dataset[0])))

        # shuffle dataset
        s_index, s_dataset = self.shuffle_dataset(index, gen_dataset)

        return s_index, s_dataset

    def shuffle_dataset(self, index, dataset):
        mapIndexPosition = list(zip(index, dataset))
        random.shuffle(mapIndexPosition)
        s_index, s_dataset = zip(*mapIndexPosition)

        print("original index = {}".format(index))
        print("shuffled index = {}".format(s_index))
        # print("shuffled dataset = {}".format(s_dataset))
        return s_index, s_dataset

