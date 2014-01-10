CC=nvcc -arch=sm_35

sum: sum.o
	$(CC) -o $@ $^

sum.o: sum.cu
	$(CC) -o $@ $< -c

clean:
	rm -f sum sum.o
