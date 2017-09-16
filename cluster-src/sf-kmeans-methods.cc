//==========================================================================// 
// Copyright 2009 Google Inc.                                               // 
//                                                                          // 
// Licensed under the Apache License, Version 2.0 (the "License");          // 
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  // 
//                                                                          //
//      http://www.apache.org/licenses/LICENSE-2.0                          //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        // 
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. // 
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
//==========================================================================//
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu   

#include "sf-kmeans-methods.h"

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <iostream>
#include <map>
#include <set>

extern int objective_output_times;
namespace sofia_cluster {

  // ---------------------------------------------------
  //         Helper functions (Not exposed in API)
  // ---------------------------------------------------

  int RandInt(int num_vals) {
    return static_cast<int>(rand()) % num_vals;
  }

  float RandFloat() {
    return static_cast<float>(rand() / static_cast<float>(RAND_MAX));
  }

  const SfSparseVector& RandomExample(const SfDataSet& data_set, int *id_x = NULL) {
    int num_examples = data_set.NumExamples();
    int i = static_cast<int>(rand()) % num_examples;
    if (i < 0) {
      i += num_examples;
    }
    if (id_x != NULL)
      *id_x = i;
    return data_set.VectorAt(i);
  }

  // ---------------------------------------------------
  //          Kmeans Initialization Functions
  // ---------------------------------------------------

  void InitializeWithKRandomCenters(int k,
                                    const SfDataSet& data_set,
                                    SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    std::set<int> selected_centers;
    // Sample k centers uniformly at random, with replacement.
    for (int i = 0; i < k; ++i) {
      cluster_centers->AddClusterCenterAt(RandomExample(data_set));
    }
  }

  void SamplingFarthestFirst(int k,
			     int sample_size,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point.
    int id = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(id));
    // Get the next k - 1 points.
    int center_id;
    for (int i = 1; i < k; ++i) {
      int best_distance = 0;
      int best_center = 0;
      for (int j = 0; j < sample_size; ++j) {
	int temp_id = RandInt(data_set.NumExamples());
	float temp_distance = cluster_centers->
	  SqDistanceToClosestCenter(data_set.VectorAt(temp_id), &center_id);
	if (temp_distance > best_distance) {
	  best_distance = temp_distance;
	  best_center = temp_id;
	}
      }
      cluster_centers->AddClusterCenterAt(data_set.VectorAt(best_center));
    }
  }

  void ClassicKmeansPlusPlus(int k,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point.
    int id = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(id));
    // Get the next k - 1 points.
    for (int i = 1; i < k; ++i) {
      // First, compute the total distance-mass, and distance for each point.
      float total_distance_mass = 0.0;
      std::map<float, int> distance_for_points;
      int center_id;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance =
	  cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j),
						     &center_id);
	if (distance > 0) {
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
    }
  }

  void OptimizedKmeansPlusPlus(int k,
                               const SfDataSet& data_set,
                               SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));
    vector<float> best_center_ids(data_set.NumExamples(), 0);
    vector<float> best_distances(data_set.NumExamples(), FLT_MAX);
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      best_distances[i] =
	cluster_centers->SqDistanceToCenterId(0, data_set.VectorAt(i));
    }

    // Get the next (k - 1) points.
    for (int i = 1; i < k; ++i) {
      float total_distance_mass = 0.0;
      std::map<float, int> distance_for_points;
      int recently_added_center = i - 1;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance =
	  cluster_centers->SqDistanceToCenterId(recently_added_center,
						data_set.VectorAt(j));
	if (distance < best_distances[j]) {
	  best_distances[j] = distance;
	  best_center_ids[j] = recently_added_center;
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	} else {
	  distance_for_points[best_distances[j] + total_distance_mass] = j;
	  total_distance_mass += best_distances[j];
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
    }
  }

  void OptimizedKmeansPlusPlusTI(int k,
				 const SfDataSet& data_set,
				 SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));
    vector<float> best_center_ids(data_set.NumExamples(), 0);
    vector<float> best_distances(data_set.NumExamples());
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      best_distances[i] =
	cluster_centers->SqDistanceToCenterId(0, data_set.VectorAt(i));
    }

    vector<float> inter_center_distances;
    // Get the next (k - 1) points.
    for (int i = 1; i < k; ++i) {
      float total_distance_mass = 0.0;
      int recently_added_center = i - 1;
      std::map<float, int> distance_for_points;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance;
	if (i >= 2 &&
	    inter_center_distances[best_center_ids[j]] >
	    2.0 * best_distances[j]) {
	  distance = best_distances[j];
	} else {
	  distance =
	    cluster_centers->SqDistanceToCenterId(recently_added_center,
						  data_set.VectorAt(j));
	}
	if (distance < best_distances[j]) {
	  best_distances[j] = distance;
	  best_center_ids[j] = recently_added_center;
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	} else {
	  distance_for_points[best_distances[j] + total_distance_mass] = j;
	  total_distance_mass += best_distances[j];
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      // Add the new cluster center and update the inter-cluster distances.
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
      inter_center_distances.clear();
      for (int j = 0; j < cluster_centers->Size() - 1; ++j) {
	inter_center_distances.push_back(cluster_centers->
	  SqDistanceToCenterId(j,
			       data_set.VectorAt(distance_iter->second)));
      }
    }
  }

  void SamplingKmeansPlusPlus(int k,
                              int sample_size,
                              const SfDataSet& data_set,
                              SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    assert(sample_size > 0);
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));

    int cluster_id;
    for (int i = 1; i < k; ++i) {
      int selected_center = 0;
      float total_distance_mass = 0.0;
      for (int j = 0; j < sample_size; ++j) {
	int proposed_cluster_center = RandInt(data_set.NumExamples());
	float distance = cluster_centers->SqDistanceToClosestCenter(
	  data_set.VectorAt(proposed_cluster_center),
	  &cluster_id);
	total_distance_mass += distance;
	if (RandFloat() < distance / total_distance_mass) {
	  selected_center = proposed_cluster_center;
	}
      }
      cluster_centers->AddClusterCenterAt(data_set.VectorAt(selected_center));
    }

  }

  // ---------------------------------------------------
  //          Kmeans Optimization Functions
  // ---------------------------------------------------

  void ProjectToL1Ball(float L1_lambda,
		       float L1_epsilon,
		       SfClusterCenters* cluster_centers) {
    if (L1_lambda > 0) {
      for (int i = 0; i < cluster_centers->Size(); ++i) {
	if (L1_epsilon == 0.0) {
	  cluster_centers->MutableClusterCenter(i)->ProjectToL1Ball(L1_lambda);
	} else {
	  cluster_centers->MutableClusterCenter(i)->
	    ProjectToL1Ball(L1_lambda, L1_epsilon);
	}
      }
    }
  }

  void BatchKmeans(int num_iterations,
                   const SfDataSet& data_set,
                   SfClusterCenters* cluster_centers,
                   float L1_lambda,
                   float L1_epsilon) {

    char filename[200];
    sprintf(filename, "batch_t_%d.txt", num_iterations);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "batch_v_%d.txt", num_iterations);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse = 0;
	//fprintf(t, "%lf ", timeuse);
    //fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int i = 0; i < num_iterations; ++i) {
      //clock_t start = clock();
      gettimeofday(&t1,NULL);
      OneBatchKmeansOptimization(data_set, cluster_centers);
      //double num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
      gettimeofday(&t2,NULL);
      timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
      //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }
  }

  void SGDKmeans(int num_iterations,
                 const SfDataSet& data_set,
                 SfClusterCenters* cluster_centers,
                 float L1_lambda,
                 float L1_epsilon) {
    vector<int> per_center_step_counts;
    per_center_step_counts.resize(cluster_centers->Size());
    
    char filename[200];
    sprintf(filename, "sgd_t_%d.txt", num_iterations);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "sgd_v_%d.txt", num_iterations);
    FILE* v = fopen(filename, "w");

    int q = num_iterations / objective_output_times;
    int r = num_iterations % objective_output_times;
    int cnt = 0;
    int upper = 0;

    struct timeval t1,t2;
    double timeuse = 0;
    //fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int i = 0; i < num_iterations; ++i) {
      //clock_t start = clock();
      gettimeofday(&t1,NULL);
      OneStochasticKmeansStep(RandomExample(data_set),
			      cluster_centers,
			      &per_center_step_counts);
      //double num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
      gettimeofday(&t2,NULL);
      timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
	  if (i == upper) {
		fprintf(t, "%lf ", timeuse);
		fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
		int interval = cnt < r ? q+1 : q;
		if (cnt == objective_output_times-1)
			upper += interval-1;
		else
			upper += interval;
		++cnt;
	  }
	  //if (i % 100 == 50)
        //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }    
    //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
  }

  void MiniBatchKmeans(int num_iterations,
		       int mini_batch_size,
		       const SfDataSet& data_set,
		       SfClusterCenters* cluster_centers,
		       float L1_lambda,
		       float L1_epsilon) {
    vector<int> per_center_step_counts;
    per_center_step_counts.resize(cluster_centers->Size());

    char filename[200];
    sprintf(filename, "mb_t_%d_%d.txt", num_iterations, mini_batch_size);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "mb_v_%d_%d.txt", num_iterations, mini_batch_size);
    FILE* v = fopen(filename, "w");

	int q = num_iterations / objective_output_times;
	int r = num_iterations % objective_output_times;
	int cnt = 0;
	int upper = 0;

    struct timeval t1,t2;
    double timeuse = 0;
	//fprintf(t, "%lf ", timeuse);
    //fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int i = 0; i < num_iterations; ++i) {
      //clock_t start = clock();
	  gettimeofday(&t1,NULL);
      OneMiniBatchKmeansOptimization(data_set,
				     cluster_centers,
				     mini_batch_size,
				     &per_center_step_counts);
      //double num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
	  gettimeofday(&t2,NULL);
	  timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
	  if (i == upper) {
		fprintf(t, "%lf ", timeuse);
		fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
		int interval = cnt < r ? q+1 : q;
		if (cnt == objective_output_times-1)
			upper += interval-1;
		else
			upper += interval;
		++cnt;
	  }
      //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }    
    //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
  }

  void OneBatchKmeansOptimization(const SfDataSet& data_set,
          SfClusterCenters* cluster_centers) {
    assert(cluster_centers->Size() > 0);
    SfClusterCenters new_centers(cluster_centers->GetDimensionality(), cluster_centers->Size());
    vector<int> examples_per_cluster(cluster_centers->Size(), 0);

    // Sum the vectors for each center.
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
      new_centers.MutableClusterCenter(closest_center)->AddVector(data_set.VectorAt(i), 1.0);
      ++examples_per_cluster[closest_center];
    }

    // Scale each center by 1/number of vectors.
    for (int i = 0; i < cluster_centers->Size(); ++i) {
      if (examples_per_cluster[i] > 0) {
        new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / examples_per_cluster[i]);
      }
    }
    // Swap in the new centers.
    cluster_centers->Clear();
    for (int i = 0; i < new_centers.Size(); ++i) {
      cluster_centers->AddClusterCenter(new_centers.ClusterCenter(i));
    }
  }

  void OneStochasticKmeansStep(const SfSparseVector& x,
                               SfClusterCenters* cluster_centers,
                               vector<int>* per_center_step_counts) {
    // Find the closest center.
    int closest_center;
    cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
    
    // Take the step.
    float c = 1.0;
    float eta = c / (++(*per_center_step_counts)[closest_center] + c);
    cluster_centers->MutableClusterCenter(closest_center)->ScaleBy(1.0 - eta);
    cluster_centers->MutableClusterCenter(closest_center)->AddVector(x, eta);
  }
  
  void OneMiniBatchKmeansOptimization(const SfDataSet&  data_set,
              SfClusterCenters* cluster_centers,
              int mini_batch_size,
              vector<int>* per_center_step_counts) {
    // Compute closest centers for a mini-batch.
    vector<vector<int> > mini_batch_centers(cluster_centers->Size());
    for (int i = 0; i < mini_batch_size; ++i) {
      // Find the closest center for a random example.
      int x_id = RandInt(data_set.NumExamples());
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(x_id),
             &closest_center);
      mini_batch_centers[closest_center].push_back(x_id);
    }
    // Apply the mini-batch.
    for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
      for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
        float c = 1.0;
        float eta = c / (++(*per_center_step_counts)[i] + c);
        cluster_centers->MutableClusterCenter(i)->ScaleBy(1.0 - eta);
        cluster_centers->MutableClusterCenter(i)->AddVector(data_set.VectorAt(mini_batch_centers[i][j]), eta);
      }
    }
  }

  void SVRGKmeans(int num_iterations,
                  int num_m,
                  const SfDataSet& data_set,
                  SfClusterCenters* cluster_centers,
                  float eta,
                  float L1_lambda,
                  float L1_epsilon) 
  {
    //cluster_centers->UpdatesMemAlloc(data_set.NumExamples(), false);
    int *table = new int[data_set.NumExamples()];

    char filename[200];
    sprintf(filename, "svrg_t_%d_%d_%f.txt", num_iterations, num_m, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_v_%d_%d_%f.txt", num_iterations, num_m, eta);
    FILE* v = fopen(filename, "w");
	
    int q = num_iterations / objective_output_times;
    int r = num_iterations % objective_output_times;
    int cnt = 0;
    int upper = 0;

    struct timeval t1,t2;
    double timeuse = 0;
    //fprintf(t, "%lf ", timeuse);
    //fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
	
	//num_m = 0.05 * data_set.NumExamples();
    
	for (int ite = 0; ite < num_iterations; ++ite) {
      gettimeofday(&t1, NULL); 
	  
	 // num_m *= 2;

      //cluster_centers->InitUpdates();

      assert(cluster_centers->Size() > 0);
      SfClusterCenters new_centers(cluster_centers->GetDimensionality(), cluster_centers->Size());
      vector<int> examples_per_cluster(cluster_centers->Size(), 0);

      // Sum the vectors for each center.
      for (int i = 0; i < data_set.NumExamples(); ++i) {
        int closest_center;
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
        new_centers.MutableClusterCenter(closest_center)->AddVector(data_set.VectorAt(i), 1.0);
        ++examples_per_cluster[closest_center];
        table[i] = closest_center;
      }

      // Scale each center by 1/number of vectors.
      for (int i = 0; i < cluster_centers->Size(); ++i) {
        if (examples_per_cluster[i] > 0) {
          new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / examples_per_cluster[i]);
          //new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / data_set.NumExamples());
        }
      }

      // Swap in the new centers.
      cluster_centers->Clear();
      for (int i = 0; i < new_centers.Size(); ++i) {
        cluster_centers->AddClusterCenter(new_centers.ClusterCenter(i));
      }

      //cluster_centers->MB2SetAvgUpdates(new_centers, examples_per_cluster, data_set.NumExamples());
      //cluster_centers->MB2SetAvgUpdates(new_centers);
	  for (int i = 0; i < cluster_centers->Size(); ++i)
		  cluster_centers->MutableClusterCenter(i)->ScaleToOne();
/*    gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);*/

      SfClusterCenters pre_centers = *cluster_centers;

      //float** avg_updates = cluster_centers->GetAvgUpdates();
      float* pre_weights;
      float* weights;
      int id_x;
      int closest_center;

      //gettimeofday(&t1, NULL);
      for (int inner = 0; inner < num_m; ++inner) {
        // Find the closest center.
        const SfSparseVector& x = RandomExample(data_set, &id_x);
        cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
        pre_weights = pre_centers.ClusterCenter(table[id_x]).GetWeight();
        if (closest_center == table[id_x])
          cluster_centers->MutableClusterCenter(closest_center)->AddVectorCompact(/*table[id_x],*/ eta, pre_weights/*, avg_updates*/);
        else {
		  weights = cluster_centers->ClusterCenter(closest_center).GetWeight();
          cluster_centers->MutableClusterCenter(closest_center)->UpdateWeights(/*closest_center, */eta, x, weights, /*avg_updates,*/ 0);
          cluster_centers->MutableClusterCenter(table[id_x])->UpdateWeights(/*table[id_x], */eta, x, pre_weights, /*avg_updates,*/ 1);
          
        }
        /*for (int i = 0; i < cluster_centers->Size(); ++i) {
            if (i != table[id_x] && i != closest_center)
              cluster_centers->MutableClusterCenter(i)->UpdateWeights(i, eta, x, pre_weights, avg_updates, 2);  
        }*/
      }

	  gettimeofday(&t2, NULL);
	  timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
	  if (ite == upper) {
		fprintf(t, "%lf ", timeuse);
		fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
		int interval = cnt < r ? q+1 : q;
		if (cnt == objective_output_times-1)
			upper += interval-1;
		else
			upper += interval;
		++cnt;
	  }
    }   
  }

  void SAGAKmeans(int num_iterations,
                  const SfDataSet& data_set,
                  SfClusterCenters* cluster_centers,
                  float eta)
  {
    char filename[200];
    sprintf(filename, "saga_t_%d_%f.txt", num_iterations, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "saga_v_%d_%f.txt", num_iterations, eta);
    FILE* v = fopen(filename, "w");

    int q = num_iterations / objective_output_times;
    int r = num_iterations % objective_output_times;
    int cnt = 0;
    int upper = 0;

    struct timeval t1,t2;
    double timeuse = 0;

    int num_samples = data_set.NumExamples();
    int *table = new int[num_samples];
    int num_centers = cluster_centers->Size();
    int dimensionality = cluster_centers->GetDimensionality();
    float **avg_updates;
    float **base_updates;

    gettimeofday(&t1,NULL);

    //cluster_centers->UpdatesMemAlloc(num_samples, true);

    avg_updates = new float*[num_centers];
    for (int i = 0; i < num_centers; ++i)
      avg_updates[i] = new float[dimensionality];

    base_updates = new float*[num_samples];
    for (int i = 0; i < num_samples; ++i)
      base_updates[i] = new float[dimensionality];

    assert(num_centers > 0);
    SfClusterCenters new_centers(cluster_centers->GetDimensionality(), num_centers);
    vector<int> examples_per_cluster(num_centers, 0);

    // Sum the vectors for each center.
    for (int i = 0; i < num_samples; ++i) {
      int closest_center;
      const SfSparseVector& x = data_set.VectorAt(i);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      new_centers.MutableClusterCenter(closest_center)->AddVector(x, 1.0);
      ++examples_per_cluster[closest_center];
      table[i] = closest_center;
      
    }

    // Scale each center by 1/number of vectors.
    for (int i = 0; i < num_centers; ++i) {
      if (examples_per_cluster[i] > 0) {
        new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / examples_per_cluster[i]);
      }
    }

    // Swap in the new centers.
   /* cluster_centers->Clear();
    for (int i = 0; i < new_centers.Size(); ++i) {
      cluster_centers->AddClusterCenter(new_centers.ClusterCenter(i));
    }*/

    for (int i = 0; i < num_samples; ++i) {
      float* weights = cluster_centers->ClusterCenter(table[i]).GetWeight();
      for(int j =0; j < dimensionality; ++j)
        base_updates[i][j] = weights[j];

      const SfSparseVector& x = data_set.VectorAt(i);
      int num_features = x.NumFeatures();
      for (int j = 0; j < num_features; ++j) {
        base_updates[i][x.FeatureAt(j)] -= x.ValueAt(j);
      }
	}
    for (int i = 0; i < num_centers; ++i) {
      cluster_centers->MutableClusterCenter(i)->ScaleToOne();
      new_centers.MutableClusterCenter(i)->ScaleToOne();
      float* new_weights = new_centers.ClusterCenter(i).GetWeight();
      float* weights = cluster_centers->ClusterCenter(i).GetWeight();
      for (int j = 0; j < dimensionality; ++j) {
        //avg_updates_[i][j] = new_weights[j] - n[i]/(float)num_examples * weights[j];
        avg_updates[i][j] =  weights[j] - new_weights[j];
      }
    }

    gettimeofday(&t2, NULL);
    timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    fprintf(t, "%lf ", timeuse);

    //fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    float* g = new float[dimensionality];
    float* val = new float[dimensionality];
    for (int ite = 0; ite < num_iterations; ++ite) {
      gettimeofday(&t1,NULL);
      int id_x;
      int closest_center;

      const SfSparseVector& x = RandomExample(data_set, &id_x);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      float* weights = cluster_centers->ClusterCenter(closest_center).GetWeight();
      int vk = examples_per_cluster[closest_center];
      if (closest_center == table[id_x]) {
        for (int i = 0; i < dimensionality; ++i) {
          g[i] = weights[i];
        }
        int num_features = x.NumFeatures();
        for (int i = 0; i < num_features; ++i) {
          g[x.FeatureAt(i)] -= x.ValueAt(i);
        }
        for (int i = 0; i < dimensionality; ++i)
          val[i] = g[i] - base_updates[id_x][i];
        cluster_centers->MutableClusterCenter(closest_center)->SAGAUpdateWeights(eta, val);
        for (int i = 0; i < num_centers; ++i) {
            cluster_centers->MutableClusterCenter(i)->SAGAUpdateWeights(eta, avg_updates[i]);
            cluster_centers->MutableClusterCenter(i)->ComputeSquaredNorm(); 
        }
        for (int i = 0; i < dimensionality; ++i) {
          avg_updates[closest_center][i] -=  val[i] / vk;
          base_updates[id_x][i] = g[i];
        }        
      }      
      else {
        for (int i = 0; i < dimensionality; ++i) {
          g[i] = weights[i];
        }
        int num_features = x.NumFeatures();
        for (int i = 0; i < num_features; ++i) {
          g[x.FeatureAt(i)] -= x.ValueAt(i);
        }
        cluster_centers->MutableClusterCenter(closest_center)->SAGAUpdateWeights(eta, g);
        cluster_centers->MutableClusterCenter(table[id_x])->SAGAUpdateWeights(-eta, base_updates[id_x]);
        
        for (int i = 0; i < num_centers; ++i) {
          cluster_centers->MutableClusterCenter(i)->SAGAUpdateWeights(eta, avg_updates[i]);
          cluster_centers->MutableClusterCenter(i)->ComputeSquaredNorm(); 
        }

        for (int i = 0; i < dimensionality; ++i) {
          avg_updates[closest_center][i] = (float)vk/(vk+1) * avg_updates[closest_center][i] + g[i]/(vk+1);
          vk = examples_per_cluster[table[id_x]];
          avg_updates[table[id_x]][i] = (float)vk/(vk-1) * avg_updates[table[id_x]][i] - base_updates[id_x][i]/(vk-1);
          base_updates[id_x][i] = g[i];
        }

        ++examples_per_cluster[closest_center];
        --examples_per_cluster[table[id_x]];
        table[id_x] = closest_center;
      }

      gettimeofday(&t2,NULL);
      timeuse += t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      if (ite == upper) {
        fprintf(t, "%lf ", timeuse);
        fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
        int interval = cnt < r ? q+1 : q;
        if (cnt == objective_output_times-1)
          upper += interval-1;
        else
          upper += interval;
        ++cnt;
      }
    }
  }

  void SVRGSKmeans(int num_iterations,
                  int num_m,
                  const SfDataSet& data_set,
                  SfClusterCenters* cluster_centers,
                  float eta,
                  float L1_lambda,
                  float L1_epsilon) 
  {
    cluster_centers->SUpdatesMemAlloc(data_set.NumExamples(), false);
    int *table = new int[data_set.NumExamples()];

    char filename[200];
    sprintf(filename, "svrg_s_t_%d_%d_%f.txt", num_iterations, num_m, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_s_v_%d_%d_%f.txt", num_iterations, num_m, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int i = 0; i < num_iterations; ++i) {
      gettimeofday(&t1, NULL); 
      cluster_centers->SInitUpdates();
      //GetSAvgGradientFast(data_set, cluster_centers, eta);
      //OneSVRGSKmeansOptimization(data_set,
      //                          cluster_centers,
      //                          num_m,
      //                          eta);
      GetSAvgGradientCompact(data_set, cluster_centers, eta, table);
      SfClusterCenters pre_centers = *cluster_centers;

      //SfClusterCenters pre_centers;
      //for (int i = 0; i < cluster_centers->Size(); ++i)
        //pre_centers.AddClusterCenter(SfWeightVector(cluster_centers->ClusterCenter(i)));
      
      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);

      gettimeofday(&t1, NULL); 
      OneSVRGSStep(data_set, cluster_centers, pre_centers, num_m, eta, table);
      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    }    
  }

  void SVRGMBKmeans(int num_iterations,
                    int num_m,
                    int mini_batch_size,
                    const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    float eta,
                    float L1_lambda,
                    float L1_epsilon) 
  {
    long int q = data_set.NumExamples() / mini_batch_size;
    int r = data_set.NumExamples() % mini_batch_size;
    long int num_mb = r ? q+1 : q;
    cluster_centers->MBUpdatesMemAlloc(num_mb, false);
    int *table = new int[data_set.NumExamples()];

    char filename[200];
    sprintf(filename, "svrg_mb_t_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_mb_v_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int i = 0; i < num_iterations; ++i) {
      gettimeofday(&t1, NULL); 
      cluster_centers->MBInitUpdates(num_mb, false);
      //GetMBAvgGradientFast(data_set, cluster_centers, mini_batch_size, eta, q, r);
      //OneSVRGMBKmeansOptimization(data_set,
      //                                   cluster_centers,
      //                                   num_m,
      //                                   mini_batch_size,
      //                                   eta,
      //                                   num_mb,
      //                                   q,
      //                                   r);
      GetMBAvgGradientCompact(data_set, cluster_centers, mini_batch_size, eta, q, r, table);
      SfClusterCenters pre_centers = *cluster_centers;

      //SfClusterCenters pre_centers;
      //for (int i = 0; i < cluster_centers->Size(); ++i)
        //pre_centers.AddClusterCenter(SfWeightVector(cluster_centers->ClusterCenter(i)));

      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);

      gettimeofday(&t1, NULL); 
      OneSVRGMBStep(data_set,
                    cluster_centers,
                    pre_centers,
                    num_m,
                    mini_batch_size,
                    eta,
                    num_mb,
                    q,
                    r,
                    table);
      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
      //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }    
    //ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
  }

  void SVRGMB2Kmeans(int num_iterations,
                    int num_m,
                    int mini_batch_size,
                    const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    float eta,
                    float L1_lambda,
                    float L1_epsilon) 
  {
    unsigned int *table = new unsigned int[data_set.NumExamples()];
    cluster_centers->UpdatesMemAlloc(data_set.NumExamples(), false);

    char filename[200];
    sprintf(filename, "svrg_mb2_t_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_mb2_v_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    for (int ite = 0; ite < num_iterations; ++ite) {
      gettimeofday(&t1, NULL); 
      cluster_centers->InitUpdates();

      assert(cluster_centers->Size() > 0);
      SfClusterCenters new_centers(cluster_centers->GetDimensionality(), cluster_centers->Size());
      vector<int> examples_per_cluster(cluster_centers->Size(), 0);

      // Sum the vectors for each center.
      for (int i = 0; i < data_set.NumExamples(); ++i) {
        int closest_center;
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
        new_centers.MutableClusterCenter(closest_center)->AddVector(data_set.VectorAt(i), 1.0);
        ++examples_per_cluster[closest_center];
        table[i] = closest_center;
      }

      // Scale each center by 1/number of vectors.
      for (int i = 0; i < cluster_centers->Size(); ++i) {
        if (examples_per_cluster[i] > 0) {
          new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / examples_per_cluster[i]);
        }
      }
      //cluster_centers->MB2SetAvgUpdates(new_centers);

      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);

      SfClusterCenters pre_centers = *cluster_centers;
      float** avg_updates = cluster_centers->GetAvgUpdates(); 

        printf("===========initial weights==========\n");
        for (int i = 0; i < cluster_centers->Size(); ++i) {
          float* w = cluster_centers->ClusterCenter(i).GetWeight();
          for (int j = 0; j < 47237; ++j) {
            if (fabsf(w[j] - 0.0) > 0.000001)
              printf("%d:%f ", j, w[j]);
          }
          printf("\n");
        }
        printf("===========initial pre_weights==========\n");
        for (int i = 0; i < cluster_centers->Size(); ++i) {
          float* w = pre_centers.ClusterCenter(i).GetWeight();
          for (int j = 0; j < 47237; ++j) {
            if (fabsf(w[j] - 0.0) > 0.000001)
              printf("%d:%f ", j, w[j]);
          }
          printf("\n");
        }
        printf("===========initial avg_weights==========\n");
        for (int i = 0; i < cluster_centers->Size(); ++i) {
          for (int j = 0; j < 47237; ++j) {
            if (fabsf(avg_updates[i][j] - 0.0) > 0.000001)
              printf("%d:%f ", j, avg_updates[i][j]);
          }
          printf("\n");
        }

        printf("*******************\n\n");

      float* pre_weights;
      int x_id;   
      for (int inner = 0; inner < num_m; ++inner) {
        gettimeofday(&t1, NULL);
        // Compute closest centers for a mini-batch.
        vector<vector<int> > mini_batch_centers(cluster_centers->Size());
        for (int i = 0; i < mini_batch_size; ++i) {
          // Find the closest center for a random example.
          x_id = RandInt(data_set.NumExamples());
          int closest_center;
          cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(x_id), &closest_center);
          mini_batch_centers[closest_center].push_back(x_id);
        }

        SfClusterCenters temp_centers = *cluster_centers;

        // Apply the mini-batch.
        for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
          printf("centroid %d: total %d\n", i, mini_batch_centers[i].size());
          for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
            x_id = mini_batch_centers[i][j];
          //  if (table[x_id] == i) {
              pre_weights = pre_centers.ClusterCenter(i).GetWeight();
              //cluster_centers->MutableClusterCenter(i)->AddVectorCompact(i, eta, pre_weights, avg_updates);
          //  } else {
          //    cluster_centers->MutableClusterCenter(i)->AddMB2VectorCompact(i, data_set.VectorAt(i), eta, avg_updates);    
          //  }
          }
          printf("\n");
        }

        printf("===========weights==========\n");
        for (int i = 0; i < cluster_centers->Size(); ++i) {
          float* w = cluster_centers->ClusterCenter(i).GetWeight();
          float* temp = temp_centers.ClusterCenter(i).GetWeight();
          for (int j = 0; j < 47237; ++j)
            if (fabsf(w[j] - temp[j]) > 0.000001)
              printf("%d:%f ", j, w[j]);
          printf("\n");
        }
        printf("===========pre_weights==========\n");
        for (int i = 0; i < cluster_centers->Size(); ++i) {
          float* w = pre_centers.ClusterCenter(i).GetWeight();
          for (int j = 0; j < 47237; ++j) {
            if (fabsf(w[j] - 0.0) > 0.000001)
              printf("%d:%f ", j, w[j]);
          }
          printf("\n");
        }

        gettimeofday(&t2, NULL);
        timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
        fprintf(t, "%lf ", timeuse);
        fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
      }
    }     
  }

void SVRGMB3Kmeans(int num_iterations,
                    int num_m,
                    int mini_batch_size,
                    const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    float eta,
                    float L1_lambda,
                    float L1_epsilon) 
  {
    unsigned int *table = new unsigned int[data_set.NumExamples()];
    cluster_centers->UpdatesMemAlloc(data_set.NumExamples(), false);

    char filename[200];
    sprintf(filename, "svrg_mb3_t_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_mb3_v_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    
    int x_id;   
    for (int inner = 0; inner < num_m; ++inner) {
      gettimeofday(&t1, NULL);
      cluster_centers->InitUpdates();
      // Compute closest centers for a mini-batch.
      vector<vector<int> > mini_batch_centers(cluster_centers->Size());
      for (int i = 0; i < mini_batch_size; ++i) {
        // Find the closest center for a random example.
        x_id = RandInt(data_set.NumExamples());
        int closest_center;
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(x_id), &closest_center);
        mini_batch_centers[closest_center].push_back(x_id);
        cluster_centers->SetUpdates(x_id, data_set.VectorAt(x_id), closest_center, false);
        table[x_id] = closest_center;
      }
      for (int i = 0; i < cluster_centers->Size(); ++i) {
        cluster_centers->SetAvgUpdates(i, mini_batch_centers[i].size());
      }
      SfClusterCenters pre_centers = *cluster_centers;
      float** avg_updates = cluster_centers->GetAvgUpdates(); 
      float* pre_weights; 
      // Apply the mini-batch.
      for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
        for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
          pre_weights = pre_centers.ClusterCenter(table[mini_batch_centers[i][j]]).GetWeight();
      //    cluster_centers->MutableClusterCenter(i)->AddVectorCompact(i, eta, pre_weights, avg_updates);
        }
      }

      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    }    
  }

  void SVRGMB4Kmeans(int num_iterations,
                    int num_m,
                    int mini_batch_size,
                    const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    float eta,
                    float L1_lambda,
                    float L1_epsilon) 
  {
    char filename[200];
    sprintf(filename, "svrg_mb4_t_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_mb4_v_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));

    for (int inner = 0; inner < num_m; ++inner) {
      gettimeofday(&t1, NULL);
      OneSVRGMB4Step(data_set, cluster_centers, mini_batch_size);
      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    }    
  }

  void OneSVRGMB4Step(const SfDataSet& data_set,
          SfClusterCenters* cluster_centers,
          int mini_batch_size) {
    assert(cluster_centers->Size() > 0);
    SfClusterCenters new_centers(cluster_centers->GetDimensionality(), cluster_centers->Size());  
    vector<int> examples_per_cluster(cluster_centers->Size(), 0);
    int x_id; 
    int closest_center;
    // Sum the vectors for each center.
    for (int i = 0; i < mini_batch_size; ++i) {
      x_id = RandInt(data_set.NumExamples());
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(x_id), &closest_center);
      new_centers.MutableClusterCenter(closest_center)->AddVector(data_set.VectorAt(x_id), 1.0);
      ++examples_per_cluster[closest_center];
    }

    // Scale each center by 1/number of vectors.
    for (int i = 0; i < cluster_centers->Size(); ++i) {
      if (examples_per_cluster[i] > 0) {
        new_centers.MutableClusterCenter(i)->ScaleBy(1.0 / examples_per_cluster[i]);
      }
    }

    // Swap in the new centers.
    cluster_centers->Clear();
    for (int i = 0; i < new_centers.Size(); ++i) {
      cluster_centers->AddClusterCenter(new_centers.ClusterCenter(i));
    }
  }

  void SVRGMB5Kmeans(int num_iterations,
                    int num_m,
                    int mini_batch_size,
                    const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    float eta,
                    float L1_lambda,
                    float L1_epsilon) {
    vector<int> per_center_step_counts;
    per_center_step_counts.resize(cluster_centers->Size());

    char filename[200];
    sprintf(filename, "svrg_mb5_t_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* t = fopen(filename, "w");
    sprintf(filename, "svrg_mb5_v_%d_%d_%d_%f.txt", num_iterations, num_m, mini_batch_size, eta);
    FILE* v = fopen(filename, "w");

    struct timeval t1,t2;
    double timeuse;
    fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));

    for (int inner = 0; inner < num_m; ++inner) {
      if (inner < 10) {
        gettimeofday(&t1, NULL);
        OneSVRGMB4Step(data_set, cluster_centers, mini_batch_size);        
      } else {
        gettimeofday(&t1, NULL);
        OneMiniBatchKmeansOptimization(data_set,
                            cluster_centers,
                            mini_batch_size,
                            &per_center_step_counts);
      }
      gettimeofday(&t2, NULL);
      timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
      fprintf(t, "%lf ", timeuse);
      fprintf(v, "%f ", KmeansObjective(data_set, *cluster_centers));
    }

  }

  void GetAvgGradientFast(const SfDataSet& data_set, 
                             SfClusterCenters* cluster_centers, 
                             float eta, 
                             int* table)
  {
    // Compute closest centers for the data set.
    vector<vector<int> > centers(cluster_centers->Size());
    long int num_examples = data_set.NumExamples();
    for (int i = 0; i < num_examples; ++i) {
      // Find the closest center for an example.
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
      cluster_centers->SetUpdates(i, data_set.VectorAt(i), closest_center, true);
      centers[closest_center].push_back(i);
      table[i] = closest_center;
    }
    for (int i = 0; i < cluster_centers->Size(); ++i) {
      //std::cout << "examples in center"<< i << " : " << centers[i].size() << std::endl;
      cluster_centers->SetAvgUpdates(i, centers[i].size());
    }  
  }

  void OneSVRGKmeansOptimization(const SfDataSet& data_set,
                                 SfClusterCenters* cluster_centers,
                                 int num_m,
                                 float eta,
	                               //vector<int>* per_center_step_counts,
	                               int* table)
  {
    float** avg_updates = cluster_centers->GetAvgUpdates();
    float** base_updates = cluster_centers->GetBaseUpdates();
    int closest_center;
    int id_x;
    for (int i = 0; i < num_m; ++i) {
      // Find the closest center.
      const SfSparseVector& x = RandomExample(data_set, &id_x);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      cluster_centers->MutableClusterCenter(closest_center)->AddVectorAlternate(id_x, x, table[id_x], eta, avg_updates, base_updates);
    }
  }

  void GetSAvgGradientFast(const SfDataSet& data_set, 
                             SfClusterCenters* cluster_centers, 
                             float eta)
  {
    long int num_examples = data_set.NumExamples();
    for (int i = 0; i < num_examples; ++i) {
      // Find the closest center for an example.
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
      cluster_centers->SSetUpdates(i, data_set.VectorAt(i), closest_center, true);
    }
    cluster_centers->SSetAvgUpdates(num_examples);
  }

  void OneSVRGSKmeansOptimization(const SfDataSet& data_set,
                                 SfClusterCenters* cluster_centers,
                                 int num_m,
                                 float eta)
  {
    float* s_avg_updates = cluster_centers->GetMBAvgUpdates();
    float** s_base_updates = cluster_centers->GetBaseUpdates();
    int closest_center;
    int id_x;
    for (int i = 0; i < num_m; ++i) {
      // Find the closest center.
      const SfSparseVector& x = RandomExample(data_set, &id_x);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      cluster_centers->MutableClusterCenter(closest_center)->AddMBVectorAlternate(id_x, x, eta, s_avg_updates, s_base_updates);
    }
  }

  void GetMBAvgGradientFast(const SfDataSet& data_set,
                               SfClusterCenters* cluster_centers, 
                               int mini_batch_size, 
                               float eta, 
                               long int q,
                               int last_mb)
  {
    long int num_examples = data_set.NumExamples();
    int closest_center;
    for (long int i = 0; i < q; ++i) {
      int upper = (i + 1) * mini_batch_size;
      for (int j = i * mini_batch_size; j < upper; ++j) {
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j), &closest_center);
        cluster_centers->MBSetUpdates(i, data_set.VectorAt(j), closest_center, true);
      }
      //cluster_centers->MBSetAvgUpdates(i, mini_batch_size);
    }
    if (last_mb) {
      for (int j = q * mini_batch_size; j < q * mini_batch_size + last_mb; ++j) {
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j), &closest_center);
        cluster_centers->MBSetUpdates(q, data_set.VectorAt(j), closest_center, true);
      }
    }
    cluster_centers->MBSetAvgUpdates(q, mini_batch_size, last_mb, num_examples, true);
    //cluster_centers->SetAvgUpdates(i, centers[i].size());
  }

  void OneSVRGMBKmeansOptimization(const SfDataSet& data_set,
                                   SfClusterCenters* cluster_centers,
                                   int num_m,
                                   int mini_batch_size,
                                   float eta,
                                   long int num_mb,
                                   long int q,
                                   int last_mb)
  {
    float* mb_avg_updates = cluster_centers->GetMBAvgUpdates();
    float** mb_base_updates = cluster_centers->GetMBBaseUpdates();
    int closest_center;
    for (int t = 0; t < num_m; ++t) {
      int id_mb = RandInt(num_mb);
      int start = id_mb * mini_batch_size;
      int end;
      if (id_mb == q && last_mb) {
        end = start + last_mb;
      } else {
        end = start + mini_batch_size;
      }

      vector<vector<int> > mini_batch_centers(cluster_centers->Size());
      for (int i = start; i < end; ++i) {
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
        mini_batch_centers[closest_center].push_back(i);
      }

      // Apply the mini-batch.
      for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
        for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
          // Find the closest center.
          cluster_centers->MutableClusterCenter(i)->AddMBVectorAlternate(id_mb, 
                        data_set.VectorAt(mini_batch_centers[i][j]), 
                        eta, 
                        mb_avg_updates, 
                        mb_base_updates);
        }
      }
    }
  }

  //
  void GetAvgGradientCompact(const SfDataSet& data_set, 
                             SfClusterCenters* cluster_centers, 
                             float eta,
                             int* table)
  {
    // Compute closest centers for the data set.
    vector<vector<int> > centers(cluster_centers->Size());
    long int num_examples = data_set.NumExamples();
    for (int i = 0; i < num_examples; ++i) {
      // Find the closest center for an example.
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
      cluster_centers->SetUpdates(i, data_set.VectorAt(i), closest_center, false);
      centers[closest_center].push_back(i);
      table[i] = closest_center;
    }
    for (int i = 0; i < cluster_centers->Size(); ++i) {
      //std::cout << "examples in center"<< i << " : " << centers[i].size() << std::endl;
      cluster_centers->SetAvgUpdates(i, centers[i].size());
    }
  }

  void OneSVRGStep(const SfDataSet& data_set,
                   SfClusterCenters* cluster_centers,
                   SfClusterCenters &pre_centers,
                   int num_m,
                   float eta,
                   int* table)
  {
    float** avg_updates = cluster_centers->GetAvgUpdates();
    int closest_center;
    //int center;
    int id_x;
    float* pre_weights;
    for (int i = 0; i < num_m; ++i) {
      // Find the closest center.
      const SfSparseVector& x = RandomExample(data_set, &id_x);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      //pre_centers.SqDistanceToClosestCenter(x, &center);
      pre_weights = pre_centers.ClusterCenter(table[id_x]).GetWeight();
      //cluster_centers->MutableClusterCenter(closest_center)->AddVectorCompact(table[id_x], eta, pre_weights, avg_updates);
    }
  }

  void GetSAvgGradientCompact(const SfDataSet& data_set, 
                              SfClusterCenters* cluster_centers, 
                              float eta,
                              int* table)
  {
    long int num_examples = data_set.NumExamples();
    for (int i = 0; i < num_examples; ++i) {
      // Find the closest center for an example.
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i), &closest_center);
      cluster_centers->SSetUpdates(i, data_set.VectorAt(i), closest_center, false);
      table[i] = closest_center;
    }
    cluster_centers->SSetAvgUpdates(num_examples);
  }

  void OneSVRGSStep(const SfDataSet& data_set,
                    SfClusterCenters* cluster_centers,
                    SfClusterCenters &pre_centers,
                    int num_m,
                    float eta,
                    int* table)
  {
    float* s_avg_updates = cluster_centers->GetMBAvgUpdates();
    int closest_center;
    //int center;
    int id_x;
    float* pre_weights;
    for (int i = 0; i < num_m; ++i) {
      // Find the closest center.
      const SfSparseVector& x = RandomExample(data_set, &id_x);
      cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
      //pre_centers.SqDistanceToClosestCenter(x, &center);
      pre_weights = pre_centers.ClusterCenter(table[id_x]).GetWeight();
      cluster_centers->MutableClusterCenter(closest_center)->AddSVectorCompact(eta, pre_weights, s_avg_updates);
    }
  }

  void GetMBAvgGradientCompact(const SfDataSet& data_set,
                               SfClusterCenters* cluster_centers, 
                               int mini_batch_size, 
                               float eta, 
                               long int q,
                               int last_mb,
                               int* table)
  {
    long int num_examples = data_set.NumExamples();
    int closest_center;
    for (long int i = 0; i < q; ++i) {
      int upper = (i + 1) * mini_batch_size;
      for (int j = i * mini_batch_size; j < upper; ++j) {
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j), &closest_center);
        cluster_centers->MBSetUpdates(i, data_set.VectorAt(j), closest_center, false);
        table[j] = closest_center;
      }
    }
    if (last_mb) {
      for (int j = q * mini_batch_size; j < q * mini_batch_size + last_mb; ++j) {
        cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j), &closest_center);
        cluster_centers->MBSetUpdates(q, data_set.VectorAt(j), closest_center, false);
        table[j] = closest_center;
      }
    }
    cluster_centers->MBSetAvgUpdates(q, mini_batch_size, last_mb, num_examples, false);

  }

  void OneSVRGMBStep(const SfDataSet& data_set,
                                   SfClusterCenters* cluster_centers,
                                   SfClusterCenters &pre_centers,
                                   int num_m,
                                   int mini_batch_size,
                                   float eta,
                                   long int num_mb,
                                   long int q,
                                   int last_mb,
                                   int* table)
  {
    float* mb_avg_updates = cluster_centers->GetMBAvgUpdates();
    int closest_center;
    for (int t = 0; t < num_m; ++t) {
      int id_mb = RandInt(num_mb);
      int start = id_mb * mini_batch_size;
      int end;
      if (id_mb == q && last_mb) {
        end = start + last_mb;
      } else {
        end = start + mini_batch_size;
      }

      vector<vector<int> > mini_batch_centers(cluster_centers->Size());
      int dimensionality = cluster_centers->GetDimensionality();
      float* mb_base_updates = new float[dimensionality];
      float* pre_weights;
      for (int j = 0; j < dimensionality; ++j) {
        mb_base_updates[j] = 0.0;
      }

      for (int i = start; i < end; ++i) {
        SfSparseVector x = data_set.VectorAt(i);
        cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
        mini_batch_centers[closest_center].push_back(i);
        pre_weights = pre_centers.ClusterCenter(table[i]).GetWeight();
        for (int j = 0; j < dimensionality; ++j) {
          mb_base_updates[j] -= pre_weights[j];
        }
        int num_features = x.NumFeatures();
        for (int k = 0; k < num_features; ++k) {
          mb_base_updates[x.FeatureAt(k)] += x.ValueAt(k);
        }
      }

      for (int j = 0; j < dimensionality; ++j) {
        mb_base_updates[j] /= (float)(end - start);
      }

      // Apply the mini-batch.
      for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
        for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
          // Find the closest center.
          cluster_centers->MutableClusterCenter(i)->AddMBVectorCompact(data_set.VectorAt(mini_batch_centers[i][j]), 
                        eta, 
                        mb_base_updates, 
                        mb_avg_updates);
        }
      }
    }
  }
  
  // ---------------------------------------------------
  //          Kmeans Evaluation Functions
  // ---------------------------------------------------

  float KmeansObjective(const SfDataSet& data_set,
		       const SfClusterCenters& cluster_centers) {
    if (cluster_centers.Size() == 0) return FLT_MAX;
    int center_id;
    float total_sq_distance = 0.0;
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      total_sq_distance +=
	cluster_centers.SqDistanceToClosestCenter(data_set.VectorAt(i),
						  &center_id);
    }
    return total_sq_distance;
  }

}  // namespace sofia_cluster
