CC=nvcc -arch=sm_35

mpi: mpi.o
	$(CC) -o $@ $^ -L ${MVAPICH_ROOT}/lib -lmpich

mpi.o: mpi.cu
	$(CC) -o $@ $< -c -I ${MVAPICH_ROOT}/include

clean:
	rm -f mpi mpi.o
