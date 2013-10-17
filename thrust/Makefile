CC=nvcc

thrust: thrust.o
	$(CC) -o $@ $^

thrust.o: thrust.cu
	$(CC) -o $@ $< -c -I /usr/local/cuda/include

clean:
	rm -f thrust thrust.o
