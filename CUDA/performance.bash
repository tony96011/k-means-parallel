for NUM in 1000 10000 100000 1000000; do
    srun -N1 -n1 --gres=gpu:1 ./cuda.out 3 ../input/input_$NUM output_datapoints_$NUM_CUDA output_centroid_$NUM
done