import unittest

from clustering.online import OnlineHAC
from clustering.rotation import RotationHAC
from linkage.single import *
from model.data_point import TrivialDataPoint


class MyTestCase(unittest.TestCase):
    def test_Rotate(self):
        data = [
            TrivialDataPoint(-1),
            TrivialDataPoint(1),
            TrivialDataPoint(-3),
            TrivialDataPoint(2),
            TrivialDataPoint(3),
            TrivialDataPoint(-2),
        ]
        clustering = RotationHAC(single_linkage)
        for d in data:
            clustering.insert(d)
        clustering.dendrogram.print()
        self.assertTrue(True)
        print(clustering.dendrogram.to_json())


if __name__ == '__main__':
    unittest.main()
