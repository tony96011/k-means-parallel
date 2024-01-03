
INPUT_FILE = sample_dataset_5000_3.txt
# cuda compile
nvcc -std=c++11 main_CUDA.cu  -o cuda.out
# cuda run
srun -N1 -n1 --gres=gpu:1 ./cuda.out 3 $INPUT_FILE output_datapoints_cuda output_centroids_cuda

diff 