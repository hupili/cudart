NVCC=nvcc -arch=sm_35

vectorAdd: vectorAdd.o
	$(NVCC) -o $@ $^

vectorAdd.o: vectorAdd.cu
	$(NVCC) -o $@ $< -c

clean:
	rm -f vectorAdd vectorAdd.o
