CC=nvcc

cublas: cublas.o
	$(CC) -o $@ $^ -lcublas

cublas.o: cublas.cu
	$(CC) -o $@ $< -c -I ~/cuda-5.5/samples/common/inc

clean:
	rm -f cublas cublas.o
