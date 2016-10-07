#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "LatticeElement.h"

# ifndef LATTICE
# define LATTICE

#define FILL_OFFSET ((float)0.001)
#define LONELY_THRESH ((float)0.1)

#pragma once

class latticed3q19
{
private:
	int _width, _height, _depth, _stride, _numberElements;
	
	float _tau, _w, _vMax, c;

	// Values that help maintain fluid stability
	float _cellsPerSide, cellSize, viscosity, timeStep, _domainSize, gravity, latticeAcceleration;

	// Solid Boundary: This is the boundary condition for a solid node. All the f's are reversed - this is known as "bounce-back"
	void solid_BC(int i0);

	void in_BC(vector3d inVector);

	void setNeighborhoodFlags();

	float calculateMassExchange(latticeElementd3q19 currentCell, latticeElementd3q19 neighborCell, int l);

	void calculateAirEquilibriumFunction(vector3d velocity, float *feq);

	void cellFlagReinitialization();

	void setFilledOrEmptied(int i, int j, int k);

	void averageSurroundings(int i, int j, int k);

	vector3d calculateNormal(int i, int j, int k);

	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);

	//Boundary Conditions
	void applyBoundaryConditions(void);

	// Collision using Single Relaxation time BGK
	void collide(void);

	std::vector<vector3d> _filledCells, _emptiedCells;

public:
	latticeElementd3q19 *latticeElements;
	
	latticed3q19::latticed3q19(int width, int height, int depth, float worldViscosity, float mass, float cellsPerSide, float domainSize);
	~latticed3q19();

	void step(void);

	void logInterfaseValues(std::ofstream &file, std::string message);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void);
};
#endif