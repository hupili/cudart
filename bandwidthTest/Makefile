NVCC=nvcc -arch=sm_35

bandwidthTest: bandwidthTest.o
	$(NVCC) -o $@ $^

bandwidthTest.o: bandwidthTest.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f bandwidthTest bandwidthTest.o
