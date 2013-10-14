CUDA Samples
============

All samples make use of the runtime API.

vectorAdd
---------

This sample adds two vectors of float.

zeroCopy
--------

This sample uses zero copy to map a host pointer to a device pointer so that kernels can read and write directly to pinned system memory.

bandwidthTest
-------------

This sample measures host to device and device to host copy bandwidth for pageable, page-locked and write-combined memory of transfer sizes 3KB, 15KB, 15MB and 100MB, and outputs them in CSV format.

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
