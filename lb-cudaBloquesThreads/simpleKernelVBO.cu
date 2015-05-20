#pragma once

#ifndef __BASIC_INTEROP_H__
 #define __BASIC_INTEROP_H__

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

typedef struct
{
	float4 pos;
	float4 color;
	float4 dir_speed;
} Vertex;

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(Vertex* v, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output vertex
	int index = y * width + x;
	if( v[index].pos.x < -0.99f )
	{
		v[index].dir_speed.x = 1.0f;
	}
	else if( v[index].pos.x > 0.99f )
	{
		v[index].dir_speed.x = -1.0f;
	}
	v[index].pos.x += (0.001f*v[index].dir_speed.x*v[index].dir_speed.z);

	if( v[index].pos.y < -0.99f )
	{
		v[index].dir_speed.y = 1.0f;
	}
	else if( v[index].pos.y > 0.99f )
	{
		v[index].dir_speed.y = -1.0f;
	}
	v[index].pos.y += (0.001f*v[index].dir_speed.y*v[index].dir_speed.z);

	//Para verlo en 3D:
	v[index].pos.z = sin(time*v[index].dir_speed.z);
}

extern "C" void runCuda(cudaGraphicsResource** resource, Vertex* devPtr, int dim, float dt)
{
	//Getting an actual address in device memory that can be passed to our kernel. 
	//We achieve this by instructing the CUDA runtime to map the
	//shared resource and then by requesting a pointer to the mapped resource.
    checkCudaErrors( cudaGraphicsMapResources( 1, resource, NULL ) );
    // devPtr is our device memory
    size_t  size;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, *resource) );

	//launchKernel (devPtr, DIM, dt);
    dim3    numBlocks(dim/16,dim/16);
    dim3    numThreads(16,16);
    kernel<<<numBlocks,numThreads>>>( devPtr, dim, dim, dt );

	//unmapping our shared resource. This call is important to make prior to performing rendering tasks because it
	//provides synchronization between the CUDA and graphics portions of the application. Specifically, 
	//it implies that all CUDA operations performed prior to the call
	//to cudaGraphicsUnmapResources() will complete before ensuing graphics
	//calls begin.
	checkCudaErrors( cudaGraphicsUnmapResources( 1, resource, NULL ) );
}

extern "C" void unregRes(cudaGraphicsResource** res)
{
	checkCudaErrors( cudaGraphicsUnmapResources( 1, res, NULL ) );
}

extern "C" void chooseDev(int ARGC, const char **ARGV)
{
	gpuGLDeviceInit(ARGC, ARGV);
}

extern "C" void regBuffer(cudaGraphicsResource** res, unsigned int& vbo)
{
	// setting up graphics interoperability by notifying the CUDA runtime 
	//that we intend to share the OpenGL buffer named vbo with CUDA.
	checkCudaErrors( cudaGraphicsGLRegisterBuffer( res, vbo, cudaGraphicsMapFlagsWriteDiscard ) );
}

#endif
