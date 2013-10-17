CC=nvcc -arch=sm_35

asyncEngine: asyncEngine.o
	$(CC) -o $@ $^

asyncEngine.o: asyncEngine.cu
	$(CC) -o $@ $< -c

clean:
	rm -f asyncEngine asyncEngine.o
