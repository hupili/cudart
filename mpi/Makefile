CC=nvcc -arch=sm_11

mpi: mpi.o
	$(CC) -o $@ $^ -L /usr/lib/openmpi -lmpi -lmpi_cxx

mpi.o: mpi.cu
	$(CC) -o $@ $< -c

clean:
	rm -f mpi mpi.o
