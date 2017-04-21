---
title: "Introduction"
teaching: 30
exercises: 0
questions:
objectives:
keypoints:
---
## Architecture

A central processing unit (CPU) is designed to handle complex tasks, such as time slicing, virtual machine emulation, complex control flows and 
branching, security, etc. In contrast, graphical processing unites (GPUs) only do one thing well. They handle billions of repetitive low level tasks. 

Originally designed for the rendering of triangles in 3D graphics, they have thousands of arithmetic logic units (ALUs) compared with traditional 
CPUs that commonly have only 4 or 8. Many types of scientific algorithms spend most of their time doing just what GPUs are good for: performing 
billions of repetitive arithmetic operations. 

Computer scientists have been quick to harness the power of GPUs for computational science applications.

The following diagram shows how GPU performance has increased compared to traditional CPU architetures.

![floating point operations]({{ page.root }}/fig/01-flops.png "floating point operations")

The reason behind the discrepancy in floating-point capability between the CPU and the GPU is that the GPU is specialized for compute-intensive, 
highly parallel computation - exactly what graphics rendering is about - and therefore designed such that more transistors are devoted to data 
processing rather than data caching and flow control.

![CPU and GPU architecture]({{ page.root }}/fig/01-cpugpuarch.png "CPU and GPU architecture")

More specifically, the GPU is especially well-suited to address problems that can be expressed as data-parallel computations - the same program is 
executed on many data elements in parallel - with a high ratio of arithmetic operations to memory operations.

Because the same program is executed for each data element, there is a lower requirement for sophisticated flow control, and because it is executed 
on many data elements and has high arithmetic intensity, the memory access latency can be hidden with calculations instead of big data caches.

### Data-parallel processing maps data elements to parallel processing threads

Many applications that process large data sets can use a data-parallel programming model to speed up the computations. In 3D rendering, large 
sets of pixels and vertices are mapped to parallel threads. Similarly, image and media processing applications such as post-processing of 
rendered images, video encoding and decoding, image scaling, stereo vision, and pattern recognition can map image blocks and pixels to 
parallel processing threads. In fact, many algorithms outside the field of image rendering and processing are accelerated by data-parallel 
processing, from general signal processing or physics simulation to computational finance or computational biology.

The advent of multicore CPUs and manycore GPUs means that mainstream processor chips are now parallel systems. Furthermore, their 
parallelism continues to scale with Moore's law. The challenge is to develop application software that transparently scales its 
parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their 
parallelism to manycore GPUs with widely varying numbers of cores.

![CPU and GPU cores]({{ page.root }}/fig/01-cpugpucores.png "CPU and GPU cores")

### Difference between a CPU and a GPU

[![Difference between a CPU and a GPU](https://img.youtube.com/vi/-P28LKWTzrI/0.jpg)](https://www.youtube.com/watch?v=-P28LKWTzrI)

## Programming

When computer scientists first attempted to use GPUs for scientific computing, the scientific codes had to be mapped onto the matrix operations for 
manipulating traingles. This was incredibly difficult to do, and took a lot of time and dedication. However, there are now high level languages 
(such as CUDA and OpenCL) that target the GPUs directly, so GPU programming is rapidly becoming mainstream in the scientific community.

A GPU program comprises two parts: a host part the runs on the CPU and one or more kernels that run on the GPU. Typically, the CPU portion of 
the program is used to set up the parameters and data for the computation, while the kernel portion performs the actual computation. In some 
cases the CPU portion may comprise a parallel program that performs message passing operations using MPI.

![GPU programming]({{ page.root }}/fig/01-programming.png "GPU programming")

