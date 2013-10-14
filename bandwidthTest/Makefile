CC=nvcc -arch=sm_35

bandwidthTest: bandwidthTest.o
	$(CC) -o $@ $^

bandwidthTest.o: bandwidthTest.cu
	$(CC) -o $@ $< -c

clean:
	rm -f bandwidthTest bandwidthTest.o
