CC=nvcc -arch=sm_35

matrixMul: matrixMul.o
	$(CC) -o $@ $^

matrixMul.o: matrixMul.cu
	$(CC) -o $@ $< -c

clean:
	rm -f matrixMul matrixMul.o
