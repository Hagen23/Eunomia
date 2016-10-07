#include <iostream>
#include <vector>
#include <cmath>

#pragma once

# ifndef LATTICE_ELEMENT
# define LATTICE_ELEMENT

//Macro to linearly go over the arrays
#define I3D(width, height,i,j,k) width*(j+height*k)+i

#pragma region utilities
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

struct vector3d
{
	float x, y, z;

	vector3d()
	{
		x = y = z = 0.0;
	}

	vector3d(float xIn, float yIn, float zIn)
	{
		x = xIn; y = yIn; z = zIn;
	}

	inline vector3d& operator +=(const vector3d& rh)
	{
		x += rh.x;
		y += rh.y;
		z += rh.z;
		return *this;
	}

	inline vector3d operator+(const vector3d& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline vector3d& operator -=(const vector3d& rh)
	{
		x -= rh.x;
		y -= rh.y;
		z -= rh.z;
		return *this;
	}

	inline vector3d operator-(const vector3d& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline vector3d& operator *=(const float& rh)
	{
		x *= rh;
		y *= rh;
		z *= rh;
		return *this;
	}

	inline vector3d operator*(const float& rhs)
	{
		*this *= rhs;
		return *this;
	}

	inline vector3d& operator /=(const float& rh)
	{
		x /= rh;
		y /= rh;
		z /= rh;
		return *this;
	}

	inline vector3d operator/(const float& rhs)
	{
		*this /= rhs;
		return *this;
	}

	bool operator ==(const vector3d& rh)
	{
		return (x == rh.x && y == rh.y && z == rh.z);
	}

	bool operator !=(const vector3d& rh)
	{
		return (x != rh.x && y != rh.y && z != rh.z);
	}

	void printVector()
	{
		std::cout << x << " " << y << " " << z << std::endl;
	}

	float dotProduct(vector3d vectorIn)
	{
		return x*vectorIn.x + y*vectorIn.y + z * vectorIn.z;
	}
};

static inline float dot(vector3d a, vector3d b)
{
	return a.x*b.x + a.y*b.y + a.z * b.z;
}

static inline vector3d float3_ScalarMultiply(const float s, const vector3d v)
{
	return vector3d{ s*v.x, s*v.y, s*v.z, };
}

static inline float float3_Norm(const vector3d v)
{
	return sqrtf(dot(v, v));
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
static vector3d speedDirection[19] =
{
	{ 0, 0, 0 },
	{ 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 },
	{ 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { 1, -1, 0 },
	{ 1, 0, 1 }, { 1, 0, -1 }, { -1, 1, 0 }, { -1, -1, 0 },
	{ -1, 0, 1 }, { -1, 0, -1 }, { 0, 1, 1 }, { 0, 1, -1 },
	{ 0, -1, 1 }, { 0, -1, -1 }
};

// The speeds for the distribution functions
static vector3d inverseSpeedDirection[19] =
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
#pragma endregion

class latticeElementd3q19
{
	friend class latticed3q19;

private:
	int		_vectorVelocitiesSize;
	float	epsilon;
	float	ro, rovx, rovy, rovz, v_sq_term; //v_sq_term is the 1.5 u^2 term of the equilibrium distribution function

	void calculateRo(void);

	float getEpsilon(void);

	vector3d getNormal();

public:
	float *f, *feq;	// Arrays to store the f values, and feq values
	float *ftemp;	//Temporal array to store the f values
	
	float cellMass, cellMassTemp;

	float c; // Speed of sound of the lattice

	int cellType, cellTypeTemp;

	bool isSolid;

	vector3d velocityVector;

	latticeElementd3q19(void);
	
	~latticeElementd3q19(void);

	void printElement(void);

	void calculateQuantities(void);
	
	void calculateEquilibriumFunction(vector3d inVector, float inRo);

	void calculateInEquilibriumFunction(vector3d inVector, float inRo);

	float getRo(void)
	{
		return ro;
	}
};

#endif