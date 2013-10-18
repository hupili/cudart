CC=nvcc -arch=sm_35 -Xcompiler -rdynamic

memcheckDemo: memcheckDemo.o
	$(CC) -o $@ $^

memcheckDemo.o: memcheckDemo.cu
	$(CC) -o $@ $< -c

clean:
	rm -f memcheckDemo memcheckDemo.o
