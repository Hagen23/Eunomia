#include <iostream>
#include <vector>

#include "LatticeElement.h"

# ifndef LATTICE
# define LATTICE

#pragma once

class latticed3q19
{
private:
	int _width, _height, _depth, _numberElements;
	float _tau;

	double c;
	
	// Dirichlet and Neumann Boundary Conditions
	void boundary_BC(vector3d inVector);

	void top_bottom_boundary(void);
	void left_right_boundary(void);
	void front_back_boundary(void);

	// Solid Boundary: This is the boundary condition for a solid node. All the f's are reversed - this is known as "bounce-back"
	void solid_BC(int i0);

	void in_BC(vector3d inVector);

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);

	//Boundary Conditions
	void applyBoundaryConditions(void);

	// Collision using Single Relaxation time BGK
	void collide(void);

public:
	latticeElementd3q19 *latticeElements;
	
	latticed3q19(int width, int height, int depth, float tau);	
	~latticed3q19();

	void step(void);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void);
};
#endif