import re
from typing import Tuple, List

import numpy as np
import glob
import imageio

from model.cluster import GroundTruthCluster
from model.data_point import BinaryVectorDataPoint, DataPoint


def generate_realworld_dataset(dirname='aloi-500-balance', shuffle=None) -> \
        Tuple[List[DataPoint], List[GroundTruthCluster]]:
    return data_wrapper(raw_real_dataset(dirname, shuffle))


def raw_real_dataset(dirname='aloi-500-balance', shuffle=None) -> Tuple[List[int], List[List[int]]]:
    """
    This function returns a tuple, in which two element are two list objects.
    The second list contains all vectors which we will use as data set later.
    The first list is a list of cluster_id of the vectors in the second list,
    identifying to which cluster each vector belongs.
    """
    imageset = []
    index = []
    extension = '*.png'
    for image_path in glob.glob("{}/*/{}".format(dirname, extension)):
        cls_index = re.findall(r'(\d*)_', image_path.split('\\')[-1])
        image = imageio.imread(image_path)
        image = np.ndarray.tolist(np.reshape(image, image.size))
        imageset.append(image)
        index.append(int(cls_index[0]))
        # print('index = {}'.format(cls_index))
        # print(image_path)

    assert len(index) == len(imageset)
    print('Read {} images.'.format(len(imageset)))
    print('Dimension of image: {}.'.format(len(imageset[0])))

    if shuffle is None:
        return index, imageset

    clusters = []
    for i in range(len(index)):
        if len(clusters) <= index[i] - 1:
            while len(clusters) < index[i] - 1 + 1:
                clusters.append([])
        clusters[index[i] - 1].append(imageset[i])

    return shuffle(clusters)


# this data wrapper is need because the data structure of the output of generating dataset is different from
# what we need to cluster.
def data_wrapper(dataset) -> Tuple[List[DataPoint], List[GroundTruthCluster]]:
    n_cluster = 0
    for index in dataset[0]:
        if index >= n_cluster:
            n_cluster = index + 1
    count = [0 for i in range(n_cluster)]
    cc = [[] for i in range(n_cluster)]
    data_stream = []
    for index, cluster in enumerate(dataset[0]):
        count[cluster] += 1
        dp = BinaryVectorDataPoint(dataset[1][index], str(cluster) + "-" + str(count[cluster]))
        cc[cluster].append(dp)
        data_stream.append(dp)
    clusters = []
    for c in cc:
        clusters.append(GroundTruthCluster(c))
    return data_stream, clusters
