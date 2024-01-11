#!/bin/bash
for NUM in 1000 10000 100000 500000 1000000 2000000; do
    srun -N1 -n1 --gres=gpu:1 ./main 5 ../input/input_$NUM.txt output_datapoints_.txt output_centroid_.txt
done

rm output_datapoints_*


## test for different N, K 
##  

