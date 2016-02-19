#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>

#pragma once

# ifndef LATTICE
# define LATTICE

using namespace std;

//Macro to linearly go over the arrays
#define I3D(width, height,i,j,k)				width*(j+height*k)+i

//Macro to go over arrays that have a stride other than 1
#define I3D_S(width, height, stride, i,j,k,l)	(width*(j+height*k)+i)*stride+l

//#define FILL_OFFSET			((float)0.003)
#define FILL_OFFSET			((float)0.01)
#define LONELY_THRESH		((float)0.1)

struct float3
{
	float x, y, z;

	float3& operator +=(const float3& rh)
	{
		x += rh.x;
		y += rh.y;
		z += rh.z;
		return *this;
	}
	
	inline float3 operator+(const float3& rhs)
	{
		*this += rhs;
		return *this;
	}

	bool operator ==(const float3& rh)
	{
		return (x == rh.x && y == rh.y && z == rh.z);
	}
};

// The different states that each cell could become
enum cell_types{
	gas = 1 << 0,
	fluid = 1 << 1,
	interphase = 1 << 2,
	solid = 1 << 3,
	CT_NO_FLUID_NEIGH = 1 << 4,
	CT_NO_EMPTY_NEIGH = 1 << 5,
	CT_NO_IFACE_NEIGH = 1 << 6,
	CT_IF_TO_FLUID = 1 << 7,
	CT_IF_TO_EMPTY = 1 << 8
};

static inline float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

static inline float3 float3_ScalarMultiply(const float s, const float3 v)
{
	return float3{ s*v.x, s*v.y, s*v.z, };
}

static inline float float3_Norm(const float3 v)
{
	//// have to change to 'fabs' for 'typedef double real'
	//float a = fabsf(v.x), b = fabsf(v.y), c = fabsf(v.z);

	//if (a < b)
	//{
	//	if (b < c)
	//		return c*sqrtf(1 + sqrtf(a / c) + sqrtf(b / c));
	//	else	// a < b, c <= b
	//		return b*sqrtf(1 + sqrtf(a / b) + sqrtf(c / b));
	//}
	//else	// b <= a
	//{
	//	if (a < c)
	//		return c*sqrtf(1 + sqrtf(a / c) + sqrtf(b / c));
	//	else	// b <= a, c <= a
	//	{
	//		if (a != 0)
	//			return a*sqrtf(1 + sqrtf(b / a) + sqrtf(c / a));
	//		else
	//			return 0;
	//	}
	//}

	return sqrtf(dot(v,v));
}

static float latticeWeights[19] =
{
	1.f / 9.f,
	1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
};

// The speeds for the distribution functions
static float3 speedDirection[19] =
{
	{0, 0, 0},
	{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
	{0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {1, -1, 0},
	{1, 0, 1}, {1, 0, -1}, {-1, 1, 0}, {-1, -1, 0},
	{-1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, 1, -1},
	{0, -1, 1}, {0, -1, -1}
};

// The inverse speed indices
static int inverseSpeedDirectionIndex[19] =
{
	0, 
	2, 1, 4, 3, 
	6, 5, 12, 11, 
	14, 13, 8, 7,
	10, 9, 18, 17,
	16, 15
};

class latticed3q19
{
private:

	// The lattice dimensions
	int				_width, _height, _depth, _stride; 
	 
	int				_numberAllElements,						// The number of dfs for the entire lattice.
					_numberLatticeElements;					// The number of cells

	float			*ro, rovx, rovy, rovz, v_sq_term;

	// Values that help maintain fluid stability
	float			_cellsPerSide, cellSize, viscosity, timeStep, _domainSize, gravity, latticeAcceleration;

	// f* are the dfs for the lattice. Epsilon is the fluid fraction of a given cell
	float			*f, *ftemp, *feq, *epsilon;
	
	// W -> relaxation time; Values (0..2]; tending to 0 = more viscous
	float			_tau, c, _w, _vMax;

	// Mass of the entire fluid, and mass of single cells of the fluid.
	float			_mass, *cellMass, *cellMassTemp;

	// Lists that contain the cells that either filled or emptied. 
	vector<float3>	_filledCells, _emptiedCells;
	
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
	
	void calculateEquilibriumFunction(float3 inVector, float inRo);

	// Calculate derived quantities density and velocity from distribution functions
	void deriveQuantities(int index);

	// Calculates epsilon: the ratio of mass vs density; e = m / ro
	float calculateEpsilon(int cellIndex);

	// Calculates a cell normal, based off the surrounding cell's epsilon
	float3 calculateNormal(int i, int j, int k);

	/// Table 4.1; to remove interfase cell artifacts
	float calculateMassExchange(int currentIndex, int neighborIndex, float currentDf, float inverse_NbFi);

	// Determines if a cell filled or emptied. Filled means that an interfase cell epsilon >= 1; emptied means epsilon <= 0
	void setFilledOrEmpty(int i, int j, int k);

	// Change the cell type in order to ensure the layer of interface cells has to be closed again, once the filled and emptied interface
	//cells have been converted into their respective types
	void cellTypeAdjustment();

	// Determine if a cell has a given type of neighbors. Used to remove artifacts
	void setNeighborhoodFlags();

	// Uses the information of the surrounding cells to initialize an interfase cell
	void averageSurroundings(int i, int j, int k);

public:

	unsigned int	*solid;
	float3			*velocityVector;
	int				*cellType, *cellTypeTemp;

	// Sets the initial mass for the lattice. This method has to be called after all the cells have a defined type
	void calculateInitialMass();
	
	// Calculates the initial df for the lattice based off the equilibrium function
	void calculateInEquilibriumFunction(int index, float3 inVector, float inRo);

	latticed3q19(int width, int height, int depth, float worldViscosity, float mass, float cellsPerSide, float domainSize);
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