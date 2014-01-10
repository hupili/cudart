CC=nvcc -arch=sm_35

conditionalCopy: conditionalCopy.o
	$(CC) -o $@ $^

conditionalCopy.o: conditionalCopy.cu
	$(CC) -o $@ $< -c

clean:
	rm -f conditionalCopy conditionalCopy.o
