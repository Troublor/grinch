import unittest

from clustering.online import OnlineHAC
from linkage.single import single_linkage
from model.data_point import TrivialDataPoint


class MyTestCase(unittest.TestCase):
    def test_onlineHAC_trivialDataPoint(self):
        data = [
            TrivialDataPoint(1),

            TrivialDataPoint(3),
            TrivialDataPoint(2),
        ]
        clustering = OnlineHAC(single_linkage)
        for d in data:
            clustering.insert(d)
        clustering.print_tree()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
