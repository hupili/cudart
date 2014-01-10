CC=nvcc -arch=sm_35

wordCount: wordCount.o
	$(CC) -o $@ $^

wordCount.o: wordCount.cu
	$(CC) -o $@ $< -c

clean:
	rm -f wordCount wordCount.o
