CC=nvcc -arch=sm_35

multiDevice: multiDevice.o
	$(CC) -o $@ $^

multiDevice.o: multiDevice.cu
	$(CC) -o $@ $< -c

clean:
	rm -f multiDevice multiDevice.o
