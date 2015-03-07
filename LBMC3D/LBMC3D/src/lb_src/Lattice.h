#include <iostream>
#include <vector>

#include "LatticeElement.h"

# ifndef LATTICE
# define LATTICE

#pragma once

class latticed2q9
{
private:
	int _width, _height, _depth, _numberElements;
	float _tau;
	
	// Solid Boundary: This is the boundary condition for a solid node. All the f's are reversed - this is known as "bounce-back"
	

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	
	//Boundary Conditions
	void applyBoundaryConditions(void);

	// Collision using Single Relaxation time BGK
	

public:
	latticeElementd2q9 *latticeElements;
	
	latticed2q9(int width, int height, int depth, float tau);	
	~latticed2q9();

	void step(void);
	void stream(void);
	void collide(void);
	void solid_BC(int i0);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void);
};
#endif