CC=nvcc -arch=sm_35

zeroCopy: zeroCopy.o
	$(CC) -o $@ $^

zeroCopy.o: zeroCopy.cu
	$(CC) -o $@ $< -c

clean:
	rm -f zeroCopy zeroCopy.o
