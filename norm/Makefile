CC=nvcc -arch=sm_35

norm: norm.o
	$(CC) -o $@ $^

norm.o: norm.cu
	$(CC) -o $@ $< -c

clean:
	rm -f norm norm.o
