#!/bin/bash
for NUM in 3 5 7 10; do
    srun -N1 -n1 --gres=gpu:1 ./main $NUM ../input/input_500000.txt output_datapoints_.txt output_centroid_.txt
done

rm output_datapoints_*


## test for different N, K 
##  

