CC=nvcc -arch=sm_35

vectorAdd: vectorAdd.o
	$(CC) -o $@ $^

vectorAdd.o: vectorAdd.cu
	$(CC) -o $@ $< -c -g -G

clean:
	rm -f vectorAdd vectorAdd.o

ptx: vectorAdd.cu
	$(CC) -o $@ $< -c -ptx
