NVFLAGS  := -std=c++14 -O3 -Xptxas="-v" -arch=sm_61 
NVFLAGS += -Xcompiler "-fopenmp -pthread -Wall -Wextra -march=native"
LDFLAGS  := -lm 

EXES     := main

alls: $(EXES)

clean:
	rm -f $(EXES)

main: main_CUDA.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?