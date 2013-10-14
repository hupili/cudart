CC=nvcc -arch=sm_35

hyperQ: hyperQ.o
	$(CC) -o $@ $^

hyperQ.o: hyperQ.cu
	$(CC) -o $@ $< -c

clean:
	rm -f hyperQ hyperQ.o
