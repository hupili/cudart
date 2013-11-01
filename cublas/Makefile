CC=nvcc

cublas: cublas.o
	$(CC) -o $@ $^ -lcublas

cublas.o: cublas.cu
	$(CC) -o $@ $< -c

clean:
	rm -f cublas cublas.o
