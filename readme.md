# Introduction
#### This is the source code of the variance reduced k-means clustering methods. It is implemented based on a packet: sofia-ml (http://code.google.com/p/sofia-ml). If this project is helpful for you, please cite our paper as follows.
```@article{Zhao:2018NEU,
 author = {Zhao, Yawei and Ming, Yuewei and Zhao, Kaikai and Liu, Xinwang and Zhu, En and  Yin, Jianping},
 title = {Large-scale k-means clustering via variance reduction},
 journal = {Elsevier Neurocomputing},
 volume = {12},
 number = {4},
 month = {September},
 year = {2016},
 pages = {29:1--29:26},
}
```

# Source code

The main implementation of our algorithm is in the file "cluster-src/sf-kmeans-methods.cc". We add a function in the file and implement our algorithm in it.
```
void SVRGKmeans(int num_iterations,
                int num_m,
                const SfDataSet& data_set,
                SfClusterCenters* cluster_centers,
                float eta)
```
```
num_iterations
  Number of optimization iterations to take.
num_m
  Number of optimization inner iterations to take.
data_set
  Dataset including example number and dimensionality.
cluster_centers
  Cluster centers including cluster center number.
eta
  Learning rate.
```
# Use

* Compile
```
cd /path/to/cluster-src/
make
```
Note that the configurations in *cluster-src/Makefile* should be modified for your own runtime environment. 

* Execution

The usage of this program is exactly the same as sofia-ml packet. The file, run.sh, gives an example for using the program. 
```
./sofia-kmeans --training_file demo/demo.train --test_file demo/demo.train --cluster_assignments_out result.txt --random_seed 1 --k 2 --init_type optimized_kmeans_pp --opt_type svrg_kmeans --sample_size 1000 --mini_batch_size 300 --iterations 10 --m 100 --eta 0.02 --dimensionality 47697 --objective_after_init --objective_after_training
```
```
--training_file
  File to be used for training.
--test_file
  File to be used for testing.
--cluster_assignments_out
  Assign each example in the --test_file to its closest cluster center, and write these results to this file. Format of the file is <nearest center id>TAB<true label (if any)>. Default: no output file.
--random_seed
  When set to non-zero value, use this seed instead of seed from system clock. This can be useful for parameter tuning in cross-validation, as setting a fixed seed by hand forces examples to be sampled in the same order. However for actual training/test, this should never be used. Default: 0
--k
  The number of cluster centers to find. Must be set.
--init_type
  Initialization procedure for seeding the kmeans optimization. Options are:
  random          random selection of cluster centers
  kmeans_pp       kmeans++ initialization method (naive)
  optimized_kmeans_pp   optimized kmeans++
  Default: random
--opt_type
  Optimization procedure for kmeans objective. Options are: batch_kmeans, sgd_kmeans, mini_batch_kmeans, svrg_kmeans
Default: mini_batch_kmeans
--sample_size
  When using sampling_kmeans_pp, the number of examples to sample on each round. Default: 1000
--mini_batch_size
  "When using mini_batch_kmeans, the number of examples to sample on each round. Default: 100
--iterations
  Number of optimization iterations to take. Default: 1000
--m
  Number of optimization inner iterations to take. Default: 1000
--eta
  Learning rate. Default: 0.02
--dimensionality
  Index value of largest feature index in training data set.
--objective_after_init
  Compute value of the kmeans objective function on training data, after initializing the cluster centers. Default is not to do this.
--objective_after_training
  Compute value of the kmeans objective function on training data, after completing training the cluster centers. Default is not to do this.
```

# Numerical results
The acceleration is significant. For example, we conduct k-means clustering on the dataset: Pittsbour (http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/).

Then we obtain the more than 5x speedup. The evaluations are presented in the following figure.

 ![image](https://github.com/YaweiZhao/VRKM_sofia-ml/blob/master/images/figure_cp_pitts-1.png)


# Contact
#### Please contact us (ywming@nudt.edu.cn, zhaoyawei@nudt.edu.cn) when you have some problems about the source code.

# More details
If this project is helpful for you, please cite our paper as follows.
```@article{Zhao:2018NEU,
 author = {Zhao, Yawei and Ming, Yuewei and Zhao, Kaikai and Liu, Xinwang and Zhu, En and  Yin, Jianping},
 title = {Large-scale k-means clustering via variance reduction},
 journal = {Elsevier Neurocomputing},
 volume = {12},
 number = {4},
 month = {September},
 year = {2016},
 pages = {29:1--29:26},
}
```
