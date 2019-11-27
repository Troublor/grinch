## COMP 5331 Project @HKUST

##### Scalable Hierarchical Clustering with Tree Graft

* Type: Implementation
* Group ID: 12
* Year: 2019
* Group Member: [@ArabelaTso](https://github.com/ArabelaTso), [@Troublor](https://github.com/Troublor), [@echohermion](https://github.com/echohermion), [@Quanooo](https://github.com/QuanQuanoooooo)

---

This is an instruction about how to use our implementation of Grinch clustering algorithm. 
This project is based on the work of (Monath et al. 2019) [Scalable Hierarchical Clustering with Tree Grafting](https://dl.acm.org/citation.cfm?doid=3292500.3330929).

### Environment

Developed on Ubuntu 18.04 using Python 3.8.

Be sure to install the dependencies before executing the project code. 
```
python3 -m pip install -r requirements.txt
```

### How to use

In general, to do a HAC clustering task you need to instantiate an object of classes in clustering package (i.e. Grinch, Rotation, Online).
Insert method of the clustering object takes one DataPoint (defined in model package) object, meaning adding one more data point into dendrogram.
You can customize data point by defining a new class extending DataPoint super class. 
You may also need to provide a list of GroundTruthCluster objects to represent the clustering ground truth to facilitate dendrogram purity calculation.
A cluster consists of a list of DataPoint objects.

A skeleton of typical clustering program is like this: 
```python
from clustering.evaluation import dendrogram_purity
from clustering.grinch import Grinch
from model.cluster import GroundTruthCluster, Cluster
from model.data_point import DataPoint

# prepare the data stream and the clustering ground truth
# data point can be customized by extending the DataPoint class if you like
data_stream: List[DataPoint]
ground_truth: List[GroundTruthCluster]

# define the linkage function or make use of predefined ones in linkage package
linkage_function: Callable[[Cluster, Cluster], float]

# create clustering instance
clustering = Grinch(linkage_function)

# data stream clustering
for data_point in data_stream:
    clustering.insert(data_point)

# print dendrogram in a human-readable way
clustering.dendrogram.print()
# calculate dendrogram purity
print("dendrogram purity:", dendrogram_purity(ground_truth, clustering.dendrogram))
```

We also provide a tool to generate a synthetic dataset. It needs several parameters, as is shown in the following:
```python
from gendataset.synthetic_dataset import generate_synthetic_dataset

n_cluster: int # the number of clusters
n_datapoint: int # the number of data points in each cluster
dimension: int # the dimension of each data point
# you can use the predefined shuffle function in gendataset package
shuffle: Callable[[List[List[List[int]]]], Tuple[List[int], List[List[int]]]]

data_stream, ground_truth = generate_synthetic_dataset(n_cluster, n_datapoint, dimension, shuffle)

# do clustering...

```

### Project Structure

This project consists of several packages: 
* clustering: the clustering algorithm (i.e. Grinch, Rotation, Online) and the function to calculate dendrogram purity. 
* dendrogram: the definition of dendrogram tree. 
* model: the definition of clusters and data points used in clustering.
* linkage: some predefined linkage function
* gendataset: predefined function to generate synthetic dataset and read real dataset (ALOI) included in this project. 
* monitor: contains the dendrogram purity monitor which is used to track the dendrogram purity change during HAC clustering. Usage example can be find in ALOI_dataset_sample.py.
* nsw: the definition of nearest neighbor small world graph which is used to implement the navigable small world graph approximation of Grinch algorithm.

### Examples

In the project files, there is a test case which test that Grinch can outperform Rotation on chain-shaped clusters. 
To run the test case:
```
python3 chain_shaped_cluster_test.py
```

Also, we provide two clustering samples to show the examples how to use our implementation to do clustering on synthetic dataset and real dataset (ALOI).
`synthetic_dataset_sample.py` is the sample to use our dataset generator to generate a synthetic dataset and use our Grinch implementation to cluster.
`ALOI_dataset_sample.py` is the sample program to cluster real dataset (i.e. ALOI). They can be executed by:
```
python3 synthetic_dataset_sample.py
python3 ALOI_dataset_sample.py
```
