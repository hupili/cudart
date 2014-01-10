CC=nvcc -arch=sm_35

saxpy: saxpy.o
	$(CC) -o $@ $^

saxpy.o: saxpy.cu
	$(CC) -o $@ $< -c

clean:
	rm -f saxpy saxpy.o
