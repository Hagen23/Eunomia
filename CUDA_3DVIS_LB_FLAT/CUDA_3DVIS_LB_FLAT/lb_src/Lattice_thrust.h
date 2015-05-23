#include <iostream>
#include <stdio.h>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>

#pragma once

# ifndef LATTICE_H
# define LATTICE_H

//Macro to linearly go over the arrays
#define I3D(width, height,i,j,k)				width*(j+height*k)+i

//Macro to go over arrays that have a stride other than 1
#define I3D_S(width, height, stride, i,j,k,l)	(width*(j+height*k)+i)*stride+l

__host__ __device__
static float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

__host__ __device__
static float dotFlat(float* a, float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float latticeWeights_in[19] =
{
	1.f / 9.f,
	1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
};

static float speedDirection_in[19*3] =
{
	0, 0, 0,
	1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0,
	0, 0, 1, 0, 0, -1, 1, 1, 0, 1, -1, 0,
	1, 0, 1, 1, 0, -1, -1, 1, 0, -1, -1, 0,
	-1, 0, 1, -1, 0, -1, 0, 1, 1, 0, 1, -1,
	0, -1, 1, 0, -1, -1
};

struct latticeStream
{
	float			*f, *ftemp;
	unsigned int	width, height, depth, stride;
	unsigned int	*solid;
	float			*speedDirection;

	latticeStream(	float* _f, float *_ftemp, float* _speedDirection,
					unsigned int _width, unsigned int _height, unsigned int _depth, unsigned int _stride, 
					unsigned int *_solid):
					f(_f), ftemp(_ftemp), speedDirection(_speedDirection),
					width(_width), height(_height), depth(_depth), stride(_stride), 
					solid(_solid)
	{

	}

	template <typename Tuple>
	__device__ void operator()(Tuple t)
	{
		unsigned int iSolid, iBase, iAdvected;
		unsigned int newI, newJ, newK;
		unsigned int tx = thrust::get<0>(t);
		unsigned int ty = thrust::get<1>(t);
		unsigned int tz = thrust::get<2>(t);
		unsigned int tw = thrust::get<3>(t);
		unsigned int tw3 = tw * 3;

		iSolid = width * (ty + height * tz) + tx;

		if( !solid[iSolid] )
		{
			newI = (unsigned int)(tx + speedDirection[tw3 + 0]);
			newJ = (unsigned int)(ty + speedDirection[tw3 + 1]);
			newK = (unsigned int)(tz + speedDirection[tw3 + 2]);

			//Checking for exit boundaries
			if (newI > (width - 1)) newI = 0;
			else if (newI == 0) newI = width - 1;

			if (newJ > (height - 1)) newJ = 0;
			else if (newJ == 0) newJ = height - 1;

			if (newK > (depth - 1)) newK = 0;
			else if (newK == 0) newK = depth - 1;

			iBase = iSolid * stride + tw;
			iAdvected = (width*(newJ + height*newK) + newI)*stride + tw;
			ftemp[iBase] = f[iAdvected];
		}
	}
};

struct latticeCollide
{
	float			*f;
	float			feq[19];
	float			*velocityVector;
	float*			speedDirection;
	float*			latticeWeights;
	float			ro, rovx, rovy, rovz, tau, c;
	unsigned int	width, height, stride;
	unsigned int	*solid;

	latticeCollide(	float* _f,
					float* _velocityVector,
					float* _speedDirection,
					float* _latticeWeights,
					unsigned int _width, unsigned int _height, unsigned int _stride, float _tau, unsigned int *_solid) :
					f(_f), velocityVector(_velocityVector), speedDirection(_speedDirection), latticeWeights(_latticeWeights), width(_width), height(_height), stride(_stride), tau(_tau), solid(_solid)
	{

	}

	__device__ 
	void calculateSpeedVector(int index)
	{
		ro = rovx = rovy = rovz = 0;
		int i0 = 0;
		for (unsigned int i = 0; i<stride; i++)
		{
			i0 = index * stride + i;
			ro += f[i0];
			rovx += f[i0] * speedDirection[i * 3 + 0];
			rovy += f[i0] * speedDirection[i * 3 + 1];
			rovz += f[i0] * speedDirection[i * 3 + 2];
		}

		// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
		if (ro == ro && ro != 0.0)
		{
			velocityVector[index * 3 + 0] = rovx / ro;
			velocityVector[index * 3 + 1] = rovy / ro;
			velocityVector[index * 3 + 2] = rovz / ro;
		}
		else
		{
			velocityVector[index * 3 + 0] = 0;
			velocityVector[index * 3 + 1] = 0;
			velocityVector[index * 3 + 2] = 0;
		}
	}

	__device__
	void calculateEquilibriumFunction(unsigned int index)
	{
		float w;
		float eiU = 0;	// Dot product between speed direction and velocity
		float eiUsq = 0; // Dot product squared
		float uSq = velocityVector[index * 3] * velocityVector[index * 3] + velocityVector[index * 3 + 1] * velocityVector[index * 3 + 1] + velocityVector[index * 3 + 2] * velocityVector[index * 3 + 2];
		
		for (unsigned int i = 0; i<stride; i++)
		{
			w = latticeWeights[i];
			eiU = speedDirection[i] * velocityVector[index * 3] + speedDirection[i] * velocityVector[index * 3 + 1] + speedDirection[i] * velocityVector[index * 3 + 2];
			eiUsq = eiU * eiU;
			feq[i] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
		}
	}

	__host__ __device__
	void solid_BC(int i0)
	{
		float temp;

		temp = f[i0*stride + 1]; 	f[i0*stride + 1] = f[i0*stride + 2];		f[i0*stride + 2] = temp;		// f1	<-> f2
		temp = f[i0*stride + 3];	f[i0*stride + 3] = f[i0*stride + 4];		f[i0*stride + 4] = temp;		// f3	<-> f4
		temp = f[i0*stride + 5];	f[i0*stride + 5] = f[i0*stride + 6];		f[i0*stride + 6] = temp;		// f5	<-> f6
		temp = f[i0*stride + 7];	f[i0*stride + 7] = f[i0*stride + 12];		f[i0*stride + 12] = temp;		// f7	<-> f12
		temp = f[i0*stride + 8];	f[i0*stride + 8] = f[i0*stride + 11];		f[i0*stride + 11] = temp;		// f8	<-> f11
		temp = f[i0*stride + 9];	f[i0*stride + 9] = f[i0*stride + 14];		f[i0*stride + 14] = temp;		// f9	<-> f14
		temp = f[i0*stride + 10];	f[i0*stride + 10] = f[i0*stride + 13];		f[i0*stride + 13] = temp;		// f10	<-> f13
		temp = f[i0*stride + 15];	f[i0*stride + 15] = f[i0*stride + 18];		f[i0*stride + 18] = temp;		// f15	<-> f18
		temp = f[i0*stride + 16];	f[i0*stride + 16] = f[i0*stride + 17];		f[i0*stride + 17] = temp;		// f16	<-> f17
	}

	template <typename Tuple>
	__device__ void operator()(Tuple t)
	{
		unsigned int tx = thrust::get<0>(t);
		unsigned int ty = thrust::get<1>(t);
		unsigned int tz = thrust::get<2>(t);
		unsigned int tw = thrust::get<3>(t);

		int iBase = 0;
		int i0 = I3D(width, height, tx, ty, tz);

		if (solid[i0] == 0)
		{
			calculateSpeedVector(i0);
			calculateEquilibriumFunction(i0);

			iBase = i0 * stride + tw;
			f[iBase] = f[iBase] - (f[iBase] - feq[tw]) / tau;
		}
		else
			solid_BC(i0);
	}
};

struct latticeInEq
{
	float			*f, *ftemp;
	float3			inVector;
	float			speedDirection[19 * 3];
	float			ro, c;
	unsigned int	width, height, stride;
	float			latticeWeights[19];

	latticeInEq(float* _f, float *_ftemp, float3 _inVector,
		unsigned int _width, unsigned int _height, unsigned int _stride, float _ro, float _c) :
		f(_f), ftemp(_ftemp), inVector(_inVector),
		width(_width), height(_height), stride(_stride), ro(_ro), c(_c)
	{
		for (int i = 0; i < 19; i++)
		{
			speedDirection[i * 3 + 0] = speedDirection_in[i * 3 + 0];
			speedDirection[i * 3 + 1] = speedDirection_in[i * 3 + 1];
			speedDirection[i * 3 + 2] = speedDirection_in[i * 3 + 2];
			latticeWeights[i] = latticeWeights_in[i];
		}
	}

	template <typename Tuple>
	__device__ void operator()(Tuple t)
	{
		unsigned int tx = thrust::get<0>(t);
		unsigned int ty = thrust::get<1>(t);
		unsigned int tz = thrust::get<2>(t);
		unsigned int tw = thrust::get<3>(t);
		unsigned int tw3 = tw * 3;

		float w;
		float eiU = 0;	// Dot product between speed direction and velocity
		float eiUsq = 0; // Dot product squared
		float uSq = inVector.x * inVector.x + inVector.y * inVector.y + inVector.z * inVector.z;

		int iBase = 0;
		unsigned int index = I3D(width, height, tx, ty, tz);

		w = latticeWeights[tw];
		eiU = speedDirection[tw3] * inVector.x +speedDirection[tw3 + 1] * inVector.y + speedDirection[tw3 + 2] * inVector.z;
		eiUsq = eiU * eiU;

		iBase = (index*stride + tw);
		ftemp[iBase] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
		f[iBase] = ftemp[iBase];
	}
};

class latticed3q19
{
private:
	unsigned int							_width, _height, _depth, _stride,
											_numberAllElements, _numberLatticeElements;
	
	float									_tau, _c;

	// Stores the i, j, k, indexes
	thrust::host_vector<unsigned int>		latticeIndexes_hi;
	thrust::host_vector<unsigned int>		latticeIndexes_hj;
	thrust::host_vector<unsigned int>		latticeIndexes_hk;
	thrust::host_vector<unsigned int>		latticeIndexes_hw;

	thrust::device_vector<unsigned int>		latticeIndexes_di;
	thrust::device_vector<unsigned int>		latticeIndexes_dj;
	thrust::device_vector<unsigned int>		latticeIndexes_dk;
	thrust::device_vector<unsigned int>		latticeIndexes_dw;

	//Stores whether the lattice element is solid or not
	thrust::device_vector<unsigned int>		latticeSolidIndexes_d;

	thrust::device_vector<float>			velocityVector_d;

	//Lattice weights
	thrust::host_vector<float>				latticeWeights_h;

	//Lattice speed direction
	thrust::host_vector<float>				speedDirection_h;

	float*									speedDirection_d_ptr;
	float*									latticeWeights_d_ptr;

	//Initialized thrust variables
	void initThrust(void);

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);

	// Collision using Single Relaxation time BGK
	void collide(void);

public:

	thrust::device_vector<float>			f_d, ftemp_d, speedDirection_d, latticeWeights_d;

	thrust::host_vector<unsigned int>		latticeSolidIndexes_h;
	thrust::host_vector<float>				velocityVector_h;

	void calculateInEquilibriumFunction(float3 inVector, float inRo);
	
	latticed3q19(int width, int height, int depth, float tau);	
	~latticed3q19();

	void step(void);

	int getNumElements(void) 
	{ 
		return _numberLatticeElements; 
	}
};
#endif