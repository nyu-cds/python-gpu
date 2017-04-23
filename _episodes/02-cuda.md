---
title: "CUDA"
teaching: 30
exercises: 0
questions:
- "What is CUDA and how is it used for computing?"
- "What is the basic programming model used by CUDA?"
- "How are CUDA programs structured?"
- "What is the importance of memory in a CUDA program?"
objectives:
- "Learn how CUDA programs are structured to make efficient use of GPUs."
- "Learn how memory must be taken into consideration when writing CUDA programs."
keypoints:
- "CUDA is designed for a specific GPU architecture, namely NVIDIA's Streaming Multiprocessors."
- "CUDA has many programming operations that are common to other parallel programming paradigms."
- "The memory architecture is extremely important to obtaining good performance from CUDA programs."
---
In November 2006, NVIDIA introduced CUDA, which originally stood for "Compute Unified Device Architecture", a general purpose parallel computing 
platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more 
efficient way than on a CPU.

The CUDA parallel programming model has three key abstractions at its core:
- a hierarchy of thread groups
- shared memories
- barrier synchronization

There are exposed to the programmer as a minimal set of language extensions.

In parallel programming, granularity means the amount of computation in relation to communication (or transfer) of data. Fine-grained 
parallelism means individual tasks are relatively small in terms of code size and execution time. The data is transferred among processors 
frequently in amounts of one or a few memory words. Coarse-grained is the opposite in that data is communicated infrequently, after larger 
amounts of computation.

The CUDA abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task 
parallelism. They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by 
blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.

A kernel is executed in parallel by an array of threads:
- All threads run the same code.
- Each thread has an ID that it uses to compute memory addresses and make control decisions.

![Thread Blocks]({{ page.root }}/fig/02-threadblocks.png "Thread Blocks")

Threads are arranged as a grid of thread blocks:
- Different kernels can have different grid/block configuration
- Threads from the same block have access to a shared memory and their execution can be synchronized

![Grid]({{ page.root }}/fig/02-threadgrid.png "Grid")

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. This independence r
equirement allows thread blocks to be scheduled in any order across any number of cores, enabling programmers to write code that scales with the 
number of cores. Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to 
coordinate memory accesses.

The grid of blocks and the thread blocks can be 1, 2, or 3-dimensional.

![Thread Mapping]({{ page.root }}/fig/02-threadmapping.png "Thread Mapping")

The CUDA architecture is built around a scalable array of multithreaded *Streaming Multiprocessors (SMs)* as shown below. Each SM has a set of 
execution units, a set of registers and a chunk of shared memory.

![Streaming Multiprocessors]({{ page.root }}/fig/02-sm.png "Streaming Multiprocessors")

In an NVIDIA GPU, the basic unit of execution is the *warp*. A warp is a collection of threads, 32 in current implementations, that are executed 
simultaneously by an SM. Multiple warps can be executed on an SM at once.

When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to SMs with available execution 
capacity. The threads of a thread block execute concurrently on one SM, and multiple thread blocks can execute concurrently on one SM. As thread 
blocks terminate, new blocks are launched on the vacated SMs.

The mapping between warps and thread blocks can affect the performance of the kernel. It is usually a good idea to keep the size of a thread block 
a multiple of 32 in order to avoid this as much as possible.

### Thread Identity

The index of a thread and its *thread ID* relate to each other as follows:

- For a 1-dimensional block, the thread index and thread ID are the same
- For a 2-dimensional block, the thread index (x,y) has thread ID=x+yD<sub>x</sub>, for block size (D<sub>x</sub>,D<sub>y</sub>)
- For a 3-dimensional block, the thread index (x,y,x) has thread ID=x+yD<sub>x</sub>+zD<sub>x</sub>D<sub>y</sub>, for 
block size (D<sub>x</sub>,D<sub>y</sub>,D<sub>z</sub>)

When a kernel is started, the number of blocks per grid and the number of threads per block are fixed (`gridDim` and `blockDim`). CUDA makes 
four pieces of information available to each thread:

- The thread index (`threadIdx`)
- The block index (`blockIdx`)
- The size and shape of a block (`blockDim`)
- The size and shape of a grid (`gridDim`)

Typically, each thread in a kernel will compute one element of an array. There is a common pattern to do this that most CUDA programs use are shown
below.

#### For a 1-dimensional grid:

~~~
tx = cuda.threadIdx.x
bx = cuda.blockIdx.x
bw = cuda.blockDim.x
i = tx + bx * bw
array[i] = compute(i)
~~~
{: .python}

#### For a 2-dimensional grid:

~~~
tx = cuda.threadIdx.x
ty = cuda.threadIdx.y
bx = cuda.blockIdx.x
by = cuda.blockIdx.y
bw = cuda.blockDim.x
bh = cuda.blockDim.y
x = tx + bx * bw
y = ty + by * bh
array[x, y] = compute(x, y)
~~~
{: .python}

### Memory Hierarchy

The CPU and GPU have separate *memory spaces*. This means that data that is processed by the GPU must be moved from the CPU to the GPU before 
the computation starts, and the results of the computation must be moved back to the CPU once processing has completed.

#### Global memory

This memory is accessible to all threads as well as the host (CPU).

- Global memory is allocated and deallocated by the host
- Used to initialize the data that the GPU will work on

![Global Memory]({{ page.root }}/fig/02-globalmemory.png "Global Memory")

#### Shared memory

Each *thread block* has its own shared memory

- Accessible only by threads within the block
- Much faster than local or global memory
- Requires special handling to get maximum performance
- Only exists for the lifetime of the block

![Shared Memory]({{ page.root }}/fig/02-sharedmemory.png "Shared Memory")

#### Local memory

Each *thread* has its own private local memory

- Only exists for the lifetime of the thread
- Generally handled automatically by the compiler

![Local Memory]({{ page.root }}/fig/02-localmemory.png "Local Memory")

#### Constant and texture memory

These are read-only memory spaces accessible by all threads.

- Constant memory is used to cache values that are shared by all functional units
- Texture memory is optimized for texturing operations provided by the hardware
