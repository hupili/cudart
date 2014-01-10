CC=nvcc -arch=sm_35

arbitraryTransformation: arbitraryTransformation.o
	$(CC) -o $@ $^

arbitraryTransformation.o: arbitraryTransformation.cu
	$(CC) -o $@ $< -c

clean:
	rm -f arbitraryTransformation arbitraryTransformation.o
