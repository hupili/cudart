CC=nvcc -arch=sm_35

conditionalCopy: conditionalCopy.o
	$(CC) -o $@ $^

conditionalCopy.o: conditionalCopy.cu
	$(CC) -o $@ $< -c

clean:
	rm -f conditionalCopy conditionalCopy.o

omp:
	g++ -I${CUDA_ROOT}/include -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP conditionalCopy.cpp -fopenmp

