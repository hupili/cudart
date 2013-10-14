CC=nvcc -arch=sm_35

atomicAdd: atomicAdd.o
	$(CC) -o $@ $^

atomicAdd.o: atomicAdd.cu
	$(CC) -o $@ $< -c -I~/cuda-5.5/samples/common/inc

clean:
	rm -f atomicAdd atomicAdd.o
