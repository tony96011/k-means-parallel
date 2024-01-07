#!/bin/bash
for NUM in 500000; do
    srun -N1 -n1 --gres=gpu:1 ./main 20 ../input/input_$NUM.txt output_datapoints_.txt output_centroid_.txt
done

rm output_datapoints_*


## test for different N, K 
##  

