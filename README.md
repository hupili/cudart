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

This sample uses atomic operations and printf.

hyperQ
------

This sample uses multiple streams to exploit the HyperQ technology.

deviceQuery
-----------

This sample enumerates the properties of the CUDA devices present in the system.
