#include <iostream>
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <ctime>

#pragma once

# ifndef LATTICE
# define LATTICE

using namespace std;

//Macro to linearly go over the arrays
#define I3D(width, height,i,j,k)				width*(j+height*k)+i

//Macro to go over arrays that have a stride other than 1
#define I3D_S(width, height, stride, i,j,k,l)	(width*(j+height*k)+i)*stride+l

static float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
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
	int				_width, _height, _depth, _stride, _numberAllElements, _numberLatticeElements;

	float			ro, rovx, rovy, rovz, v_sq_term;

	float			*f, *ftemp, *feq;
	
	float			_tau, c;
	
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

	void calculateSpeedVector(int index);

	void calculateEquilibriumFunction(int index);

	

public:

	unsigned int	*solid;
	float3			*velocityVector;

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