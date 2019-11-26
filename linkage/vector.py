from typing import List

import numpy as np

from model.cluster import Cluster


def vector_cosine(v1: List[int], v2: List[int]):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_similarity_for_binary_vector_data_point(c1: Cluster, c2: Cluster) -> float:
    # the cosine similarity of two clusters
    v1 = list(map(lambda elem: elem.vector, c1.data_points))
    v2 = list(map(lambda elem: elem.vector, c2.data_points))
    return vector_cosine(np.sum(v1, axis=0), np.sum(v2, axis=0))
