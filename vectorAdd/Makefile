CC=nvcc -arch=sm_35

vectorAdd: vectorAdd.o
	$(CC) -o $@ $^

vectorAdd.o: vectorAdd.cu
	$(CC) -o $@ $< -c

clean:
	rm -f vectorAdd vectorAdd.o
