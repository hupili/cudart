CC=nvcc

deviceQuery: deviceQuery.o
	$(CC) -o $@ $^

deviceQuery.o: deviceQuery.cu
	$(CC) -o $@ $< -c

clean:
	rm -f deviceQuery deviceQuery.o
