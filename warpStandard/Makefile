CC=nvcc

warpStandard: warpStandard.o
	$(CC) -o $@ $^

warpStandard.o: warpStandard.cu
	$(CC) -o $@ $< -c -gencode arch=compute_11,code=sm_11 -gencode arch=compute_35,code=sm_35

clean:
	rm -f warpStandard warpStandard.o
