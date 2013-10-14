CC=nvcc -arch=sm_35

atomicAdd: atomicAdd.o
	$(CC) -o $@ $^

atomicAdd.o: atomicAdd.cu
	$(CC) -o $@ $< -c

clean:
	rm -f atomicAdd atomicAdd.o
