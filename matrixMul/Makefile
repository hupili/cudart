NVCC=nvcc -arch=sm_35

matrixMul: matrixMul.o
	$(NVCC) -o $@ $^

matrixMul.o: matrixMul.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f matrixMul matrixMul.o
