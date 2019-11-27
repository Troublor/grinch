from model.cluster import Cluster

"""
Files in this package includes some linkage functions. 

A linkage function should take two Cluster objects as input parameters and output a float value, 
indicating the similarity of two clusters. 
The larger the output value is, the more similar the two clusters are.
"""


def chain_linkage_for_trivial_data_point(c1: Cluster, c2: Cluster) -> float:
    # if there is not edge between any two node in two clusters, then the similarity is -1
    # otherwise the similarity is 1
    for dp1 in c1.data_points:
        for dp2 in c2.data_points:
            if dp1.pairwise_similarity(dp2) <= 1:
                return 1
    else:
        return -1
