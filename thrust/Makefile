CC=nvcc

thrust: thrust.o
	$(CC) -o $@ $^

thrust.o: thrust.cu
	$(CC) -o $@ $< -c -I ~/cuda-5.5/include

clean:
	rm -f thrust thrust.o
