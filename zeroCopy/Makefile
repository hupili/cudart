NVCC=nvcc -arch=sm_35

zeroCopy: zeroCopy.o
	$(NVCC) -o $@ $^

zeroCopy.o: zeroCopy.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f zeroCopy zeroCopy.o
