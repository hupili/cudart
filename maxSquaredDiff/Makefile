CC=nvcc -arch=sm_35

maxSquaredDiff: maxSquaredDiff.o
	$(CC) -o $@ $^

maxSquaredDiff.o: maxSquaredDiff.cu
	$(CC) -o $@ $< -c

clean:
	rm -f maxSquaredDiff maxSquaredDiff.o
