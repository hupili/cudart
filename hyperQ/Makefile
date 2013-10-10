NVCC=nvcc -arch=sm_35

hyperQ: hyperQ.o
	$(NVCC) -o $@ $^

hyperQ.o: hyperQ.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f hyperQ hyperQ.o
