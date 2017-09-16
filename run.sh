#!/bin/bash
./sofia-kmeans --training_file demo/demo.train --test_file demo/demo.train --cluster_assignments_out result.txt --random_seed 1 --k 2 --init_type optimized_kmeans_pp --opt_type svrg_mb_kmeans --sample_size 1000 --mini_batch_size 300 --iterations 10 --m 100 --eta 0.02 --dimensionality 47697 --objective_after_init --objective_after_training
