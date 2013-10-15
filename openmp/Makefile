CC=nvcc -arch=sm_35 -Xcompiler -fopenmp

openmp: openmp.o
	$(CC) -o $@ $^

openmp.o: openmp.cu
	$(CC) -o $@ $< -c -I ~/cuda-5.5/samples/common/inc

clean:
	rm -f openmp openmp.o
