#include <iostream>
#include <vector>

# ifndef LATTICE_ELEMENT
# define LATTICE_ELEMENT

#pragma once

//Macro to linearly go over the arrays
#define I3D(width, height,i,j,k) width*(j+height*k)+i

struct vector3d
{
	double x, y, z;

	vector3d()
	{
		x = y = z = 0.0;
	}

	vector3d(double xIn, double yIn, double zIn)
	{
		x = xIn; y = yIn; z = zIn;
	}

	void printVector()
	{
		std::cout << x << " " << y << " " << z << std::endl;
	}

	double dotProduct(vector3d vectorIn)
	{
		return x*vectorIn.x + y*vectorIn.y + z * vectorIn.z;
	}
};

static vector3d speedDirection[19] =
{
	vector3d(0,0,0),
	vector3d(1,0,0),	vector3d(0,1,0),	vector3d(-1,0,0),	vector3d(0,-1,0), 	
	vector3d(0,0,-1),	vector3d(0,0,1),	vector3d(1,1,0),	vector3d(-1,1,0),	
	vector3d(-1,-1,0),	vector3d(1,-1,0),	vector3d(1,0,-1),	vector3d(-1,0,-1),	
	vector3d(-1,0,1),	vector3d(1,0,1),	vector3d(0,1,-1),	vector3d(0,1,1),	
	vector3d(0,-1,-1),	vector3d(0,-1,1)
};

class latticeElementd3q19
{
private:
	int _vectorVelocitiesSize;
	double ro, rovx, rovy, rovz, v_sq_term; //v_sq_term is the 1.5 u^2 term of the equilibrium distribution function

	void calculateRo(void);

public:
	double *f, *feq;	// Arrays to store the f values, and feq values
	double *ftemp;	//Temporal array to store the f values
	
	bool isSolid;

	vector3d velocityVector;

	latticeElementd3q19(void);
	
	~latticeElementd3q19(void);

	void printElement(void);

	void calculateSpeedVector(void);
	
	void calculateEquilibriumFunction();

	void calculateInEquilibriumFunction(vector3d inVector, float inRo);
};

#endif