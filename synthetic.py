from typing import List

from gendataset.generate_dataset import DataGeneration
from model.data_point import DataPoint


def data_wrapper(dataset: List) -> List[DataPoint]:
    pass


gen = DataGeneration()
output = gen.gen_random_dataset(n_cluster=2, n_point_each_cluster=3, n_dim_datapoint=3)
print(output)
