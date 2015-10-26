# ifndef LATTICE_H
# define LATTICE_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <helper_cuda.h> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

//Macro to linearly go over the arrays
__host__ __device__
#define I3D(width, height,i,j,k)				width*(j+height*k)+i

//Macro to go over arrays that have a stride other than 1
__host__ __device__
#define I3D_S(width, height, stride, i,j,k,l)	(width*(j+height*k)+i)*stride+l

__host__ 
static float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

__host__ __device__
static float dot(float ax, float ay, float az, float bx, float by, float bz)
{
	return ax * bx + ay * by + az * bz;
}

static float latticeWeights[19] =
{
	1.f / 9.f,
	1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
};

static float3 speedDirection[19] =
{
	{0, 0, 0},
	{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
	{0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {1, -1, 0},
	{1, 0, 1}, {1, 0, -1}, {-1, 1, 0}, {-1, -1, 0},
	{-1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, 1, -1},
	{0, -1, 1}, {0, -1, -1}
};

class latticed3q19
{
private:
	// These describe the lattice configuration
	int				_width, _height, _depth, _stride, _numberAllElements, _numberLatticeElements;

	// Macroscopic fluid density
	float			ro, rovx, rovy, rovz, v_sq_term;

	// f - Particle distribution function
	float			*f, *ftemp, *feq, *f_d, *ftemp_d;

	// Macroscopic velocity
	float			*velocityVector_d;
	
	// _tau - relaxation time, elementary time of collisions; c - Basic speed of the lattice
	float			_tau, c;
	
	unsigned int 	*solid_d;

	std::ofstream		outputFile;

	// Dirichlet and Neumann Boundary Conditions
	void boundary_BC(float3 inVector);

	// Solid Boundary: This is the boundary condition for a solid node. All the f's are reversed - this is known as "bounce-back"
	void solid_BC(int i0);

	void in_BC(float3 inVector);

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);

	//Boundary Conditions
	void applyBoundaryConditions(void);

	// Collision using Single Relaxation time BGK
	void collide(void);

	// Streaming and collide steps that will use shared memory
	void stream_collide(void);

	void calculateSpeedVector(int index);

	void calculateEquilibriumFunction(int index);

	void initCUDA();
	

public:

	unsigned int	*solid;
	float			*velocityVector;

	void calculateInEquilibriumFunction(int index, float3 inVector, float inRo);
	
	latticed3q19(int width, int height, int depth, float tau);	
	~latticed3q19();

	void step(void);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void) 
	{ 
		return _numberLatticeElements; 
	}
};
#endif
