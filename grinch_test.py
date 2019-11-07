import unittest

from clustering.grinch import Grinch
from linkage.single import *
from model.data_point import TrivialDataPoint


class MyTestCase(unittest.TestCase):
    # def test_approximations(self):
    #     data = [
    #         TrivialDataPoint(-1),
    #         TrivialDataPoint(1),
    #         TrivialDataPoint(-3),
    #         TrivialDataPoint(2),
    #         TrivialDataPoint(3),
    #         TrivialDataPoint(-2),
    #     ]
    #     clustering = Grinch(single_linkage, navigable_small_world_graphs=True, k_nsw=5, capping=True, capping_height=2, single_elimination=True, single_nn_search=True, k_nn=3)
    #     for d in data:
    #         print("insert", d)
    #         clustering.insert(d)
    #         clustering.dendrogram.print()
    #     self.assertTrue(True)

    def test_normal(self):
        data = [
            TrivialDataPoint(-1),
            TrivialDataPoint(1),
            TrivialDataPoint(-3),
            TrivialDataPoint(2),
            TrivialDataPoint(3),
            TrivialDataPoint(-2),
        ]
        clustering = Grinch(single_linkage)
        for d in data:
            print("insert", d)
            clustering.insert(d)
            clustering.dendrogram.print()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
