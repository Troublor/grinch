import random
from typing import List, Tuple

"""
This file defines different shuffle functions to shuffle the dataset (the order of data stream).
 
A shuffle function should take a List of clusters as input parameter. 
Each cluster is a List of vectors and each vector is a List of integer.

A shuffle function should also output a Tuple, in which two element are two list objects. 
The second list contains all vectors which we will use as data set later.
The first list is a list of cluster_id of the vectors in the second list, 
identifying to which cluster each vector belongs.  
"""


def round_robin(clusters: List[List[List[int]]]) -> Tuple[List[int], List[List[int]]]:
    K = len(clusters)
    c = [i for i in range(K)]
    random.shuffle(c)
    i = [0 for _ in range(K)]

    count = 0
    for cluster in clusters:
        count += len(cluster)
    data_stream = []
    index = []
    for j in range(count):
        c_id = j % K
        index.append(c_id)
        data_stream.append(clusters[c[c_id]][i[c_id]])
        i[c_id] += 1
    return index, data_stream


def sorted_shuffle(clusters: List[List[List[int]]]) -> Tuple[List[int], List[List[int]]]:
    K = len(clusters)
    c = [i for i in range(K)]
    index = []
    data_stream = []
    for j in range(K):
        index += [c[j] for _ in range(len(clusters[c[j]]))]
        data_stream += clusters[c[j]]
    return index, data_stream


def random_shuffle(clusters: List[List[List[int]]]) -> Tuple[List[int], List[List[int]]]:
    index = []
    for i in range(len(clusters)):
        index += [i for _ in range(len(clusters[i]))]
    dataset = []
    for cluster in clusters:
        dataset += cluster
    mapIndexPosition = list(zip(index, dataset))
    random.shuffle(mapIndexPosition)
    s_index, s_dataset = zip(*mapIndexPosition)
    return s_index, s_dataset
