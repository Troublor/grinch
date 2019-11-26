import unittest

from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from clustering.online import OnlineHAC
from clustering.rotation import RotationHAC
from linkage.chain_shaped import chain_linkage_for_trivial_data_point
from linkage.classic import *
from model.cluster import GroundTruthCluster
from model.data_point import TrivialDataPoint

data = [
    TrivialDataPoint(-2),
    TrivialDataPoint(1),
    TrivialDataPoint(-4),
    TrivialDataPoint(-1),
    TrivialDataPoint(3),
    TrivialDataPoint(-3),
    TrivialDataPoint(2),
]
ground_truth = [
    GroundTruthCluster([data[0], data[2], data[3], data[5]]),
    GroundTruthCluster([data[1], data[6], data[4]])
]


class ChainShapedClusterTest(unittest.TestCase):
    def test_online(self):
        clustering = OnlineHAC(chain_linkage_for_trivial_data_point)
        for d in data:
            clustering.insert(d)
        self.assertTrue(dendrogram_purity(ground_truth, clustering.dendrogram) < 1)

    def test_rotation(self):
        clustering = RotationHAC(chain_linkage_for_trivial_data_point)
        for d in data:
            # print("inserting", d)
            clustering.insert(d)
            # print dendrogram to debug
            # clustering.dendrogram.print()
        # Rotation algorithm fails to achieve a perfect dendrogram purity when clustering chain-shaped clusters
        self.assertTrue(dendrogram_purity(ground_truth, clustering.dendrogram) < 1)

    def test_grinch(self):
        clustering = Grinch(chain_linkage_for_trivial_data_point)
        for d in data:
            # print("inserting", d)
            clustering.insert(d)
            # print dendrogram to debug
            # clustering.dendrogram.print()
        # Grinch algorithm can achieve a perfect dendrogram purity when clustering chain-shaped clusters.
        self.assertTrue(dendrogram_purity(ground_truth, clustering.dendrogram) == 1)


if __name__ == '__main__':
    unittest.main()
