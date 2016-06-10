#ifndef LATTICE_2D
#define LATTICE_2D

#include "Utilities.h"
#include <stdio.h>

#define SIZE_2D_X			32					//!< x-dimension of the 2D field, must be a power of 2
#define SIZE_2D_Y			32					//!< y-dimension of the 2D field, must be a power of 2
#define MASK_2D_X			(SIZE_2D_X - 1)		//!< bit mask for fast modulo 'SIZE_2D_X' operation
#define MASK_2D_Y			(SIZE_2D_Y - 1)		//!< bit mask for fast modulo 'SIZE_2D_Y' operation

#pragma region Lattice_constants

/// Bit masks for the various cell types and neighborhood flags
enum cellType
{
	CT_OBSTACLE = 1 << 0,
	CT_FLUID = 1 << 1,
	CT_INTERFACE = 1 << 2,
	CT_EMPTY = 1 << 3,
	// neighborhood flags, OR'ed with actual cell type
	CT_NO_FLUID_NEIGH = 1 << 4,
	CT_NO_EMPTY_NEIGH = 1 << 5,
	CT_NO_IFACE_NEIGH = 1 << 6,
	//to track the change of state
	CT_IF_TO_FLUID = 1 << 7,
	CT_IF_TO_EMPTY = 1 << 8
	// changing the maximum value here requires adapting the temporary cell types in 'UpdateTypesLBMStep(...)'
};

/// Velocity vectors for D2Q9 as integers
static const int2 vel2Di[9] = {
	{ 0, 0 },		// zero direction
	{ -1, 0 },		// 4 directions with velocity 1
	{ 1, 0 },
	{ 0, -1 },
	{ 0, 1 },
	{ -1, -1 },		// 4 directions with velocity sqrt(2)
	{ -1, 1 },
	{ 1, -1 },
	{ 1, 1 },
};

static const float2 vel2Dv[9] = {
	{ 0, 0 },		// zero direction
	{ -1, 0 },		// 4 directions with velocity 1
	{ 1, 0 },
	{ 0, -1 },
	{ 0, 1 },
	{ -1, -1 },		// 4 directions with velocity sqrt(2)
	{ -1, 1 },
	{ 1, -1 },
	{ 1, 1 },
};

/// Index of inverse direction
static const int invVel2D[9] = { 0, 2, 1, 4, 3, 8, 7, 6, 5 };

static const float weights2D[9] = { (float)4. / 9,
(float)1. / 9, (float)1. / 9, (float)1. / 9, (float)1. / 9,
(float)1. / 36, (float)1. / 36, (float)1. / 36, (float)1. / 36 };

#pragma endregion

/// Class that contains the information of a cell
class d2q9_cell
{
	public:
		float	f[9], ftemp[9], mex[9];
		int		type;
		float	rho;
		float	mass, mass_temp;
		float2	velocity;

		d2q9_cell();
		d2q9_cell(float *f_, float mass_ , float rho_, int type_ , float2 velocity_ );

		void deriveQuantities(float vMax_);
		float calculateEpsilon();
};

class d2q9_lattice
{
	public:
		d2q9_cell	cells[SIZE_2D_X][SIZE_2D_Y];
		float		tau, c, w, vMax;
		int			width, height, stride;
		float		cellsPerSide, cellSize, viscosity, timeStep, domainSize, gravity, latticeAcceleration;

		d2q9_lattice(int width_, int height_, float worldViscosity_, float cellsPerSide_, float domainSize_);
		
		inline d2q9_cell* getCellAt(const int x_, const int y_)
		{
			//printf("X:%d, Y:%d\n", x_, y_);
			return &cells[x_][y_];	
		}

		inline d2q9_cell* getCellAt_Mod(const int x_, const int y_)
		{
			//printf("Mod X:%d, Y:%d\n", x_, y_);
			return getCellAt(x_ & MASK_2D_X, y_ & MASK_2D_Y);
		}

		inline void print_fluid_amount()
		{
			int counter = 0;
			float global_mass_f = 0.f, global_mass_i = 0.f;

			for (int i = 0; i < width - 1; i++)
			{
				for (int j = 0; j < height - 1; j++)
				{
					d2q9_cell* current_cell = getCellAt(i, j);
					if (current_cell->type == cellType::CT_FLUID)
					{
						counter++;
						global_mass_f += current_cell->mass;
					}
					if (current_cell->type == cellType::CT_INTERFACE)
					{
						global_mass_i += current_cell->mass;
					}
				}
			}
			printf("Fluid count %d; global mass f %f; global mass i %f; Total %f \n", counter, global_mass_f, global_mass_i,
				global_mass_f+global_mass_i);
		}

		void step(void);

		void stream(void);
		void collide(void);

		/// Calculates the equilibrium function based on a velocity and density, and stores it in feq
		void calculateEquilibrium(float2 velocity_, float rho_, float feq_[9]);

		float2 calculateNormal(int x_, int y_);

		// Updates the cells' type and 'moves' the fluid
		void updateCells();

		// Averages the surrounding rho and u of an empty cell that turned to interfase
		void averageSurroundings(d2q9_cell* cell_, int x_, int y_);

		/// Calculate mass exchange so that undesired interfase cells fill or empty, mainly to avoid visual artifacts
		/// Table given in reference thesis
		//void calculateMassExchange(int current_cell_type, int nb_cell_type, );

		void initCells(int **typeArray_, float initRho_, float2 initVelocity_);
};
#endif