#include <iostream>
#include <vector>

# ifndef LATTICE
# define LATTICE

#pragma once

//Function to linearly go over the arrays
#define I3D(width, height,i,j,k) width*(j+height*k)+i

static const int speedDirection[19][3] =
{
	{1,0,0},	{-1,0,0},	{0,1,0},	{0,-1,0}, 
	{1,1,0},	{-1,-1,0},	{1,-1,0},	{-1,1,0},
	{1,0,1},	{-1,0,-1},	{1,0,-1},	{-1,0,1},
	{0,1,1},	{0,-1,-1},	{0,1,-1},	{0,-1,1},
	{0,0,1},	{0,0,-1},	{0,0,0}
};

struct vector3d
{
	float x, y, z;
};

class latticeElementd3q19
{
private:
	int _vectorVelocitiesSize;
	float ro, rovx, rovy, rovz, v_sq_term; //v_sq_term is the 1.5 u^2 term of the equilibrium distribution function

	void calculateRo(void)
	{
		ro = 0; 
		for(int i = 0; i<_vectorVelocitiesSize; i++)
			ro += f[i];
	}

public:
	float *f;
	vector3d velocityVector;

	latticeElementd3q19()
	{
		_vectorVelocitiesSize = 19;
		
		f = new float[_vectorVelocitiesSize];

		velocityVector.x = 1;
		velocityVector.y = 1;
		velocityVector.z = 1;
		
		for(int i = 0; i<_vectorVelocitiesSize; i++)
			f[i] = 0;
	}
	
	~latticeElementd3q19(void)
	{
		std::cout << "destroying velocities" << std::endl;
		delete[] f;
	}

	void printElement(void)
	{
		for(int i =0; i<_vectorVelocitiesSize; i++)
			std::cout << "f[" << i << "]: " << f[i] << " vx: " << velocityVector.x << " vy: " << velocityVector.y << " vz: " << velocityVector.z << std::endl;
	}

	void calculateSpeedVector(void)
	{
		calculateRo();
		rovx = rovy = rovz = 0; 

		for(int i=0; i<_vectorVelocitiesSize; i++)
		{
			rovx += f[i] * speedDirection[i][0];
			rovy += f[i] * speedDirection[i][1];
			rovz += f[i] * speedDirection[i][2];
		}

		velocityVector.x = rovx / ro;
		velocityVector.y = rovy / ro;
		velocityVector.z = rovz / ro;
	}
};

class latticed3q19
{
private:
	int _width, _height, _depth, _numberElements;
	
	// Perimeter Boundary: All the f's leaving the bottom of the domain (j=0) enter at the top (j=nj-1), and vice-verse
	void per_BC(void);

	// Solid Boundary: This is the boundary condition for a solid node. All the f's are reversed - this is known as "bounce-back"
	void solid_BC(void);

	// This inlet BC is extremely crude but is very stable: We set the incoming f values to the equilibirum values assuming: ro=roout; vx=vxin; vy=0
	void in_BC(void);

	// This is the very simplest (and crudest) exit BC. All the f values pointing into the domain at the exit (ni-1) are set equal to those one node into
	// the domain (ni-2)
	void ex_BC_crude(void);

public:
	latticeElementd3q19 *latticeElements, *tempLatticeElements;
	
	latticed3q19(int width, int height, int depth);	
	~latticed3q19();

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);

	//Boundary Conditions
	void applyBoundaryConditions(void);

	// Collisions between the particles are modeled here. We use the very simplest
	// model which assumes the f's change toward the local equlibrium value (based
	// on density and velocity at that point) over a fixed timescale, tau	
	void collide(void);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void);
};
#endif