#### Note: the sequential, omp, p-threads code are fork from https://github.com/vinayak1998/Parallel-K-Means-Clustering. 
#### We optimize the code using CUDA on top of it.
# Parallel-K-Means-Clustering with CUDA
- Greatly speedup k-means clustering computation time for up to 74.46x in our experiment.
- Reduce the time complexity from O(N*K*T) to O(T(K+a)), a and K are small
- In our implementation, the program is no longer bounded by computation time.
For more detail introduction, implementation and evaluation, please refer to [**PP_final_project.pdf**](https://github.com/tony96011/k-means-parallel/blob/master/PP_final_project.pdf)

## Execution
- **CUDA**
  - cd into CUDA folder
  - make to compile
  - we offer 3 script:
    - correctness.sh for testing the correctness
    - k_perf.sh for testing diff K
    - performance.sh for testing diff N
    - please refer to the bash file for sample execution
- **Input**
  - we also offer an input generator in Input folder
  - pleaes use ./input_generator N for generating the input data.   
- for other version please refer to the root author's repository
 

