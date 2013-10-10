NVCC=nvcc

deviceQuery: deviceQuery.o
	$(NVCC) -o $@ $^

deviceQuery.o: deviceQuery.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f deviceQuery deviceQuery.o
