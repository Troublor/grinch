import unittest

from clustering.grinch import Grinch
from linkage.single import *
from model.data_point import TrivialDataPoint


class MyTestCase(unittest.TestCase):
    def test_something(self):
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
        clustering.print_tree()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
