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

///// Velocity vectors for D2Q9 as integers
//static const int2 vel2Di[9] = {
//	{ 0, 0 },		// zero direction
//	{ -1, 0 },		// 4 directions with velocity 1
//	{ 1, 0 },
//	{ 0, -1 },
//	{ 0, 1 },
//	{ -1, -1 },		// 4 directions with velocity sqrt(2)
//	{ -1, 1 },
//	{ 1, -1 },
//	{ 1, 1 },
//};
//
//static const float2 vel2Dv[9] = {
//	{ 0, 0 },		// zero direction
//	{ -1, 0 },		// 4 directions with velocity 1
//	{ 1, 0 },
//	{ 0, -1 },
//	{ 0, 1 },
//	{ -1, -1 },		// 4 directions with velocity sqrt(2)
//	{ -1, 1 },
//	{ 1, -1 },
//	{ 1, 1 },
//};

/// Velocity vectors for D2Q9 as integers
static const int2 vel2Di[9] = {
	{ 0, 0 },		// 0	0
	{ 0, 1 },		// N	1
	{ 0, -1 },		// S	2
	{ 1, 0 },		// E	3
	{ -1, 0 },		// W	4
	{ 1, 1 },		// NE	5
	{ -1, 1 },		// NW	6
	{ 1, -1 },		// SE	7
	{ -1, -1 },		// SW	8
};

/// Velocity vectors for D2Q9 as integers
static const float2 vel2Dv[9] = {
	{ 0, 0 },		// 0	0
	{ 0, 1 },		// N	1
	{ 0, -1 },		// S	2
	{ 1, 0 },		// E	3
	{ -1, 0 },		// W	4
	{ 1, 1 },		// NE	5
	{ -1, 1 },		// NW	6
	{ 1, -1 },		// SE	7
	{ -1, -1 },		// SW	8
};

/// Index of inverse direction
static const int invVel2D[9] = { 0, 2, 1, 4, 3, 8, 7, 6, 5 };

/// Lattice weight for the 2d version
static const float weights2D[9] = { (float)4. / 9,
(float)1. / 9, (float)1. / 9, (float)1. / 9, (float)1. / 9,
(float)1. / 36, (float)1. / 36, (float)1. / 36, (float)1. / 36 };

#pragma endregion

/// Class that contains the information of a cell
class d2q9_cell
{
	public:

		int		type;						/// Type of the cell; the types are defined in the enum cellType
		float	f[9], ftemp[9], mex[9];		/// Distribution functions and excess mass
		float	rho;						/// Cell density
		float	mass;
		float2	velocity;					/// Macroscopic velocity

		d2q9_cell();

		/// f_			Initial distribution function for a cell -- Typically the equilibrium
		/// mass_		Initial mass for a cell
		/// rho_		Initial density for a cell
		/// type_		Initial type for a cell
		/// velocity_	Initial velocity for a cell
		d2q9_cell(float *f_, float mass_ , float rho_, int type_ , float2 velocity_ );

		/// Calculates the macroscopic quantities: density and velocity
		/// vMax_	The maximum velocity that a cell can get to. If it is exceeded, the velocity is scaled down.
		void deriveQuantities(float vMax_);

		/// Calculates the fluid fraction, epsilon = mass / density
		float calculateEpsilon();
};

class d2q9_lattice
{
	public:
		d2q9_cell	cells[SIZE_2D_X][SIZE_2D_Y];
		float		tau, c, w, vMax;				/// Single relaxation time, speed of sound, relaxation time, max fluid velocity
		int			width, height, stride;

		/// For debugging purposes
		float		total_Mass = 0.f, fluid_mass = 0.f, interface_mass = 0.f, air_mass = 0.f, solid_mass = 0.f,
			initial_mas = 0.f, initial_fluid_mass = 0.f, initial_interface_mass = 0.f, initial_air_mass = 0.f, initial_solid_mass = 0.f;

		/// Parameters to guarantee stability by considering the cell and domain sizes, as well as viscosity and gravity.
		float		cellsPerSide, cellSize, viscosity, timeStep, domainSize, gravity, latticeAcceleration;

		/// width_				The width of the lattice
		/// height_				The height of the lattice
		/// worldViscosity_		The viscosity of the fluid, m^2/s
		/// cellsPerSide_		The number of cells that are in a single side of the lattice
		/// domainSize_			The size of one side of the lattice, m - Used to define the size of each cell. May change this later.
		d2q9_lattice(int width_, int height_, float worldViscosity_, float cellsPerSide_, float domainSize_);
		
		/// Returns a cell at column x_ and row y_.
		inline d2q9_cell* getCellAt(const int x_, const int y_)
		{
			//printf("X:%d, Y:%d\n", x_, y_);
			return &cells[y_][x_];	
		}

		/// Returns a cell at column x_ and row y_. The mod operations prevents accessing a value not in the array.
		inline d2q9_cell* getCellAt_Mod(const int x_, const int y_)
		{
			//printf("Mod X:%d, Y:%d\n", x_, y_);
			return getCellAt(x_ & MASK_2D_X, y_ & MASK_2D_Y);
		}

		/// Prints the ammount of mass of the entire lattice.
		inline void print_fluid_amount(char* message = "")
		{
			int counter = 0;

			total_Mass = fluid_mass = interface_mass = air_mass = 0.f;

			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					d2q9_cell* current_cell = getCellAt(col, row);
					if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
					{
						counter++;
						fluid_mass += current_cell->mass;
					}
					else if ((current_cell->type & CT_INTERFACE)== cellType::CT_INTERFACE)
					{
						interface_mass += current_cell->mass;
					}
					else if ((current_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
					{
						air_mass += current_cell->mass;
					}
					else if ((current_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
					{
						solid_mass += current_cell->mass;
					}
				}
			}
			total_Mass = fluid_mass + interface_mass + air_mass +  solid_mass;
			//printf("Fluid count %d; f %.2f; i %.2f; e %2.f; Total %.2f \n", counter, fluid_mass, interface_mass, air_mass, total_Mass);
			if (total_Mass < (initial_mas) || (total_Mass) > initial_mas)
				printf("f %.2f; i %.2f; a %2.f; o %2.f; Total %f; %s\n df %.2f; di %.2f; dTotal %f; \n",
				fluid_mass, interface_mass, air_mass, solid_mass, total_Mass, message,
				initial_fluid_mass - fluid_mass, initial_interface_mass - interface_mass, initial_mas - total_Mass);
		}

		inline void print_types()
		{
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					d2q9_cell* current_cell = getCellAt(col, SIZE_2D_Y - row - 1);
					printf("%d", current_cell->type);
				}
				printf("\n");
			}
		}

		/// Performs a single time step calculation of the fluid.
		void step(void);

		void stream(void);
		void collide(void);

		// Updates the cells' type and 'moves' the fluid
		void updateCells();

		/// Used to determine the correct mass exchange for interface cells. Table 4.1
		/// current_type	The type of the current cell
		/// nb_type			The neighbor's cell type
		/// fi_neigh		The df of the neighbor in the direction of ei
		/// fi_inv			The inverse df in the direction of ei
		float calculateMassExchange(int current_type, int nb_type, float fi_neigh, float fi_inv);

		/// Calculates the equilibrium function based on a velocity and density, and stores it in feq.
		void calculateEquilibrium(float2 velocity_, float rho_, float feq_[9]);

		/// Calculates the normal that the fluid is forming at cell col x row y.
		float2 calculateNormal(int x_, int y_);

		// Averages the surrounding rho and u of an empty cell that turned to interfase
		void averageSurroundings(d2q9_cell* cell_, int x_, int y_);

		/// Initializes de lattice based on a predefined array of types, as well as a density and velocity for the fluid.
		void initCells(int **typeArray_, float initRho_, float2 initVelocity_);
};
#endif