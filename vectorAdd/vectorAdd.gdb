NVIDIA (R) CUDA Debugger
5.5 release
Portions Copyright (C) 2007-2013 NVIDIA Corporation
GNU gdb (GDB) 7.2
Copyright (C) 2010 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later &lt;http://gnu.org/licenses/gpl.html&gt;
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-unknown-linux-gnu".
For bug reporting instructions, please see:
&lt;http://www.gnu.org/software/gdb/bugs/&gt;...
Reading symbols from /home/hjli/cudart/vectorAdd/vectorAdd...done.
(cuda-gdb) b vectorAdd
Breakpoint 1 at 0x402dd9: file vectorAdd.cu, line 4.
(cuda-gdb) r
Starting program: ~/cudart/vectorAdd/vectorAdd 
[Thread debugging using libthread_db enabled]
[New Thread 0x7ffff6ca4700 (LWP 54366)]
[Context Create of context 0x67f380 on Device 0]
[Launch of CUDA Kernel 0 (vectorAdd<<<(196,1,1),(256,1,1)>>>) on Device 0]
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 12, warp 0, lane 0]

Breakpoint 1, vectorAdd(const float * @generic, const float * @generic, float * @generic, int)<<<(196,1,1),(256,1,1)>>> (a=0x2300300000, b=0x2300330e00, 
    c=0x2300361c00, numElements=50000) at vectorAdd.cu:5
5		int i = blockDim.x * blockIdx.x + threadIdx.x;
(cuda-gdb) p numElements
$1 = 50000
(cuda-gdb) n
6		if (i < numElements)
(cuda-gdb) p i
$2 = 0
(cuda-gdb) cuda block 1 thread 3
[Switching focus to CUDA kernel 0, grid 1, block (1,0,0), thread (3,0,0), device 0, sm 11, warp 0, lane 3]
5		int i = blockDim.x * blockIdx.x + threadIdx.x;
(cuda-gdb) n
6		if (i < numElements)
(cuda-gdb) p i
$3 = 259
(cuda-gdb) n
8			c[i] = a[i] + b[i];
(cuda-gdb) n
10	}
(cuda-gdb) n
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 12, warp 0, lane 0]
6		if (i < numElements)
(cuda-gdb) help cuda
Print or select the CUDA focus.

List of cuda subcommands:

cuda block -- Print or select the current CUDA block
cuda device -- Print or select the current CUDA device
cuda grid -- Print or select the current CUDA grid
cuda kernel -- Print or select the current CUDA kernel
cuda lane -- Print or select the current CUDA lane
cuda sm -- Print or select the current CUDA SM
cuda thread -- Print or select the current CUDA thread
cuda warp -- Print or select the current CUDA warp

Type "help cuda" followed by cuda subcommand name for full documentation.
Type "apropos word" to search for commands related to "word".
Command name abbreviations are allowed if unambiguous.
(cuda-gdb) help info cuda
Print informations about the current CUDA activities. Available options:
         devices : information about all the devices
             sms : information about all the SMs in the current device
           warps : information about all the warps in the current SM
           lanes : information about all the lanes in the current warp
         kernels : information about all the active kernels
        contexts : information about all the contexts
          blocks : information about all the active blocks in the current kernel
         threads : information about all the active threads in the current kernel
    launch trace : information about the parent kernels of the kernel in focus
 launch children : information about the kernels launched by the kernels in focus

(cuda-gdb) help set cuda
Generic command for setting gdb cuda variables

List of set cuda subcommands:

set cuda api_failures -- Set the api_failures to ignore/stop/hide on CUDA driver API call errors
set cuda break_on_launch -- Automatically set a breakpoint at the entrance of kernels
set cuda coalescing -- Turn on/off coalescing of the CUDA commands output
set cuda context_events -- Turn on/off context events (push/pop/create/destroy) output messages
set cuda defer_kernel_launch_notifications -- Turn on/off deferral of kernel launch messages
set cuda disassemble_from -- Choose whether to disassemble from the device memory (slow) or the ELF image (fast)
set cuda gpu_busy_check -- Turn on/off GPU busy check the next time the inferior application is run
set cuda hide_internal_frame -- Set hiding of the internal CUDA frames when printing the call stack
set cuda kernel_events -- Turn on/off kernel events (launch/termination) output messages
set cuda launch_blocking -- Turn on/off CUDA kernel launch blocking (effective starting from the next run)
set cuda memcheck -- Turn on/off CUDA Memory Checker next time the inferior application is run
set cuda notify -- Thread to notify about CUDA events when no other known candidate
set cuda software_preemption -- Turn on/off CUDA software preemption debugging the next time the inferior application is run
set cuda thread_selection -- Set the automatic thread selection policy to use when the current thread cannot be selected

Type "help set cuda" followed by set cuda subcommand name for full 
documentation.
Type "apropos word" to search for commands related to "word".
Command name abbreviations are allowed if unambiguous.
(cuda-gdb) del 1
(cuda-gdb) c
Continuing.
[Termination of CUDA Kernel 0 (vectorAdd<<<(196,1,1),(256,1,1)>>>) on Device 0]
[Context Destroy of context 0x67f380 on Device 0]
[Thread 0x7ffff6ca4700 (LWP 54406) exited]

Program exited normally.
