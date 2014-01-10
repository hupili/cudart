CC=nvcc -arch=sm_35

piEstimate: piEstimate.o
	$(CC) -o $@ $^

piEstimate.o: piEstimate.cu
	$(CC) -o $@ $< -c

clean:
	rm -f piEstimate piEstimate.o
