CC=nvcc

thrust: thrust.o
	$(CC) -o $@ $^

thrust.o: thrust.cu
	$(CC) -o $@ $< -c

clean:
	rm -f thrust thrust.o
