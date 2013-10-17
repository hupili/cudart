CC=nvcc

cublas: cublas.o
	$(CC) -o $@ $^ -lcublas

cublas.o: cublas.cu
	$(CC) -o $@ $< -c -I /usr/local/cuda/samples/common/inc

clean:
	rm -f cublas cublas.o
