CC=nvcc -arch=sm_35

checkError: checkError.o
	$(CC) -o $@ $^

checkError.o: checkError.cu
	$(CC) -o $@ $< -c -I /usr/local/cuda/samples/common/inc

clean:
	rm -f checkError checkError.o
