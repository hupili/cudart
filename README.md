
   * When no placeholder of `_2` is passed, degrade to `transform_reduce`.
   An existing feature?
   This is for performance sake,
   just like partial specializations in c++ std library.
   * It's said that `make_tuple` and `get` can support only 10 components.
   Then we can write a more general version using "tuple of tuple".

following original README.

-----------


CUDA Samples
============

All samples make use of the runtime API.

vectorAdd
---------

This sample adds two vectors of floats.

zeroCopy
--------

This sample maps device pointers to pinned host memory so that kernels can directly read from and write to pinned host memory.

bandwidthTest
-------------

This sample measures host-to-device and device-to-host bandwidth via PCIe for pageable and pinned memory of four transfer sizes of 3KB, 15KB, 15MB and 100MB, and outputs them in CSV format.

checkError
----------

This sample checks the return value of every runtime API.

matrixMul
---------

This sample uses shared memory to accelerate matrix multiplication.

atomicAdd
---------

This sample uses atomic functions, assertions and printf.

asyncEngine
-----------

This sample uses asynchronous engines to overlap data transfer and kernel execution.

hyperQ
------

This sample uses multiple streams to overlap multiple kernel execution, known as the HyperQ technology.

deviceQuery
-----------

This sample enumerates the properties of the CUDA devices present in the system.
multiDevice
-----------

This sample uses cudaSetDevice within a single thread to utilize multiple GPUs.

openmp
------

This sample uses OpenMP to create multiple CPU threads to utilize multiple GPUs.

mpi
---

This sample uses MPI to create multiple CPU processes to utilize multiple GPUs.

cublas
------

This sample uses CUBLAS, a CUDA implementation of BLAS (Basic Linear Algebra Subprograms), for matrix multiplication.

thrust
------

This sample uses thrust, a CUDA implementation of STL (Standard Template Library), for vector reduction.
