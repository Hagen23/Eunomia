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

#pragma region utilities
struct int3
{
	int x, y, z;

	inline int3& operator +=(const int3& rh)
	{
		x += rh.x;
		y += rh.y;
		z += rh.z;
		return *this;
	}

	inline int3 operator+(const int3& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline int3& operator -=(const int3& rh)
	{
		x -= rh.x;
		y -= rh.y;
		z -= rh.z;
		return *this;
	}

	inline int3 operator-(const int3& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline int3 operator/(const int& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline int3& operator /=(const int& rh)
	{
		x /= rh;
		y /= rh;
		z /= rh;
		return *this;
	}

	bool operator ==(const int3& rh)
	{
		return (x == rh.x && y == rh.y && z == rh.z);
	}

	bool operator !=(const int3& rh)
	{
		return (x != rh.x && y != rh.y && z != rh.z);
	}
};

struct float3
{
	float x, y, z;

	inline float3& operator +=(const float3& rh)
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

	inline float3& operator -=(const float3& rh)
	{
		x -= rh.x;
		y -= rh.y;
		z -= rh.z;
		return *this;
	}

	inline float3 operator-(const float3& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline float3 operator/(const float& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline float3& operator /=(const float& rh)
	{
		x /= rh;
		y /= rh;
		z /= rh;
		return *this;
	}

	bool operator ==(const float3& rh)
	{
		return (x == rh.x && y == rh.y && z == rh.z);
	}

	bool operator !=(const float3& rh)
	{
		return (x != rh.x && y != rh.y && z != rh.z);
	}
};

static inline float dot(int3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

static inline float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

static inline float3 float3_ScalarMultiply(const float s, const int3 v)
{
	return float3{ s*v.x, s*v.y, s*v.z, };
}

static inline float3 float3_ScalarMultiply(const float s, const float3 v)
{
	return float3{ s*v.x, s*v.y, s*v.z, };
}

static inline float float3_Norm(const float3 v)
{
	return sqrtf(dot(v,v));
}

#pragma endregion

#pragma region lattice constants
static float latticeWeights[19] =
{
	1.f / 3.f,
	1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
	1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
};

// The speeds for the distribution functions
static int3 speedDirection[19] =
{
	{0, 0, 0},
	{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
	{0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {1, -1, 0},
	{1, 0, 1}, {1, 0, -1}, {-1, 1, 0}, {-1, -1, 0},
	{-1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, 1, -1},
	{0, -1, 1}, {0, -1, -1}
};

// The speeds for the distribution functions
static float3 inverseSpeedDirection[19] =
{
	{ 0, 0, 0 },
	{ -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 },
	{ 0, 0, -1 }, { 0, 0, 1 }, { -1, -1, 0 }, { -1, 1, 0 },
	{ -1, 0, -1 }, { -1, 0, 1 }, { 1, -1, 0 }, { 1, 1, 0 },
	{ 1, 0, -1 }, { 1, 0, 1 }, { 0, -1, -1 }, { 0, -1, 1 },
	{ 0, 1, -1 }, { 0, 1, 1 }
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

enum cell_types
{
	gas, fluid, interfase, solid
};
#pragma endregion

class latticed3q19
{
private:

	// The lattice dimensions
	int				_width, _height, _depth, _stride; 
	 
	int				_numberAllElements,						// The number of dfs for the entire lattice.
					_numberLatticeElements;					// The number of cells

	// Values that help maintain fluid stability
	float			_cellsPerSide, cellSize, viscosity, timeStep, _domainSize, gravity, latticeAcceleration;

	// f* are the dfs for the lattice. Epsilon is the fluid fraction of a given cell
	float			*f, *ftemp, *ro, *cell_mass, *cell_mass_temp;
	
	// Number of fluid and gas neighbors for each cell
	int				*nFluidNbs, *nGasNbs, *nInterNbs, *nSolidNbs;
	
	// W -> relaxation time; Values (0..2]; tending to 0 = more viscous
	float			_tau, c, _w, _vMax;

	vector<int3>	_filledCells, _emptiedCells;
		
	// Move the f values one grid spacing in the directions that they are pointing
	// i.e. f1 is copied one location to the right, etc.
	void stream(void);
	
	// Collision using Single Relaxation time BGK
	void collide(void);
		
	// Add the air distributions to the interface cells, as well as reconstruct the ones coming from the direction of the 
	// interface normals
	void reconstructInterfaceDfs(void);

	// Tracks the mass that is entering and leaving the cells
	void calculateMassExchange(void);

	void calculateEquilibriumFunction(float *feq, float3 inVector, float inRo);
	
	// Calculate derived quantities density and velocity from distribution functions
	void deriveQuantities(int index);

	float getEpsilon(int cellindex);

	// Calculates how many 
	//void updateNeighborsCount(int cellIndex);

	// Determines if an interfase cell filled or emptied. Filled or emptied cells are added to queues in order to process them.
	void setFilledEmptied(int i, int j, int k);

	float3 calculateNormal(int i, int j, int k);

	void flagReinitialization();

	void averageSurroundings(int i, int j, int k);

	void distributeExcessMass();

	void setNumberNeighbors();

public:

	unsigned int	*solid;
	float3			*velocityVector;

	cell_types		*cell_type, *cell_type_temp;

	// Sets the initial mass for the lattice. This method has to be called after all the cells have a defined type
	void calculateInitialMass();

	void initLatticeDistributions();

	latticed3q19(int width, int height, int depth, float worldViscosity, float cellsPerSide, float domainSize);
	~latticed3q19();

	void step(void);

	void printLattice(void);
	void printLatticeElement(int i, int j, int k);
	int getNumElements(void) { return _numberLatticeElements; }
};
#endif