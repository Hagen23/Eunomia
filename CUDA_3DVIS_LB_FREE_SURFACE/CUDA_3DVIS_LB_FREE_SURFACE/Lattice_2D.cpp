// Sergio's
#include "Lattice_2D.h"
#include <memory>

d2q9_cell::d2q9_cell() : mass(0.0f), type(cellType::CT_EMPTY), rho(0.0f), velocity(float2{ 0.0f, 0.0f })
{
	for (int i = 0; i < 9; i++)
		f[i] = ftemp[i] = mex[i] = 0.f;
}

d2q9_cell::d2q9_cell(float *f_, float mass_ = 0.0f, float rho_ = 0.0f,
	int type_ = cellType::CT_EMPTY, float2 velocity_ = float2{ 0.0f, 0.0f })
{
	for (int i = 0; i < 9; i++)
	{
		f[i] = ftemp[i] = f_[i];
		mex[i] = 0.f;
	}

	type = type_;
	rho = rho_;
	mass = mass_;
	velocity = velocity_;
}

void d2q9_cell::deriveQuantities(float vMax_)
{
	rho = 0;
	velocity = float2{ 0.0f, 0.0f };

	for (int i = 0; i < 9; i++)
	{
		rho += f[i];
		velocity += float2_ScalarMultiply(f[i], vel2Dv[i]);
	}

	if (rho > 0.0f)
		epsilon = mass / rho;

	///// rho * v = sum(fi * ei)
	//if (rho > 0.0f)
	//{
	//	for (int i = 0; i < 9; i++)
	//		velocity += float2_ScalarMultiply(f[i], vel2Dv[i]);

	//	//velocity = float2_ScalarMultiply(1.0f / rho, velocity);
	//}

	/// Rescale in case maximum velocity is exceeded
	//float norm = float2_Norm(velocity);
	//if (norm > vMax_)
	//	velocity = float2_ScalarMultiply(vMax_ / norm, velocity);
}

float d2q9_cell::calculateEpsilon()
{
	//if ((type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
	//	return 1.f;

	//if ((type & CT_EMPTY) == cellType::CT_EMPTY)
	//	return 0.f;

	if ((type & CT_FLUID) == cellType::CT_FLUID || (type & CT_INTERFACE) == cellType::CT_INTERFACE)
	{
		if (rho > 0)
		{
			float epsilon = mass / rho;
			// df->mass can even be < 0 or > df->rho for interface cells to be converted to fluid or empty cells in the next step;
			// clamp to [0,1] range for numerical stability
			if (epsilon > 1.f)
				epsilon = 1.f;
			else if (epsilon < 0.f)
				epsilon = 0.f;

			return epsilon;
		}
		//else
		//{
		//	// return (somewhat arbitrarily) a ratio of 0.01
		//	return 0.01f;
		//}
	}

	return 0.f;
}

d2q9_lattice::d2q9_lattice(int width_, int height_, float worldViscosity_, float cellsPerSide_, float domainSize_)
{
	width = width_;
	height = height_;
	stride = 9;

	// The cells are initiated statically with the predefined size. This initialization step would be needed with dynamic pointer
	//cells = new d2q9_cell*[width];
	//for (int i = 0; i < width; i++)
	//	cells[i] = new d2q9_cell[height];

	// Values for stability and to integrate gravity
	c = (float)(1.0 / sqrt(3.0));
	domainSize = domainSize_;
	cellsPerSide = cellsPerSide_;
	cellSize = domainSize_ / cellsPerSide_;
	gravity = 9.81f;

	/// Stability concerns, section 3.3 of the reference thesis.
	/// Here a value of gc = 0.005 is used to keep the compressibility below half a percent.
	timeStep = (float)(sqrtl((0.005f * cellSize) / fabs(gravity)));

	vMax = cellSize / timeStep;
	viscosity = worldViscosity_ * timeStep / (cellSize * cellSize);
	tau = 3.0f * viscosity + 0.5f;
	w = 1.0f / tau;
	latticeAcceleration = gravity * timeStep * timeStep / cellSize;

	print_fluid_amount("Startup");
}

void d2q9_lattice::step()
{
	//collide();
	//stream();
	//updateCells();
	//print_fluid_amount();

	print_fluid_amount("Before stream");
	stream();
	print_fluid_amount("Before collide");
	collide();
	print_fluid_amount("Before update");
	updateCells();
	print_fluid_amount("After update");
	printf("\n\n");

	//print_fluid_amount("Before collide");
	//collide();
	//print_fluid_amount("After collide");
	//stream();
	//print_fluid_amount("After stream");
	//updateCells();
	//print_fluid_amount("After update");
	//printf("\n\n");

	//print_types();
}

void d2q9_lattice::transferMass()
{

}

void d2q9_lattice::stream()
{
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE && (current_cell->type & CT_EMPTY) != cellType::CT_EMPTY)
			{
				if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
				{
					for (int i = 0; i < 9; i++)
					{
						int inv = invVel2D[i];
						// f'i(x, t+dt) = fi(x+eî, t), pull scheme
						d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[inv].x, row + vel2Di[inv].y);

						/// Normal streaming, no mass exchange is needed -- or is it? Mass transfer only needs to happen
						/// from a fluid to an interface cell, not from an interface to a fluid.
						if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID ||
							(nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
						{
							//current_cell->mass += nb_cell->f[i] - current_cell->f[invVel2D[i]];
							current_cell->ftemp[i] = nb_cell->f[i];
						}
						else if ((nb_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
						{
							current_cell->ftemp[i] = current_cell->f[inv];
						}
					}
					//print_fluid_amount("After fluid df cycle");
				}
				else if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				{
					//print_fluid_amount("Before interface df cycle");
					//const float current_epsilon = current_cell->calculateEpsilon();
					const float current_epsilon = current_cell->epsilon;
					float f_atm_eq[9] = { 0 };

					/// Equilibrium calculated with air density, and the velocity of the current cell. Used to 
					/// reconstruct missing dfs.
					calculateEquilibrium(current_cell->velocity, 1.f, f_atm_eq);

					for (int i = 0; i < 9; i++)
					{
						// f'i(x, t+dt) = fi(x+eî, t)
						int inv = invVel2D[i];
						d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[inv].x, row + vel2Di[inv].y);

						if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
						{
							// dmi(x, t + dt) = fî(x+ei,t) - fi(x, t)
							// fî(x+ei,t) -> mass incoming from fluid;  fi(x, t) -> mass outgoing from interface.
							current_cell->mass += nb_cell->f[i] - current_cell->f[inv];

							/// Normal streaming
							current_cell->ftemp[i] = nb_cell->f[i];
						}
						else if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
						{
							//const float nb_epsilon = nb_cell->calculateEpsilon();
							const float nb_epsilon = nb_cell->epsilon;

							/// Eq. 4.3
							//current_cell->mass += (nb_cell->f[i] - current_cell->f[invVel2D[i]]) * 0.5f * (current_epsilon + nb_epsilon);
							current_cell->mass +=
								calculateMassExchange(current_cell->type, nb_cell->type, nb_cell->f[i], current_cell->f[inv]) *
								0.5f * (current_epsilon + nb_epsilon);

							current_cell->ftemp[i] = nb_cell->f[i];
						}
						else if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
						{
							/// no mass exchange from or to empty cell
							/// DFs that would come out of the empty cells need to be reconstructed from the boundary conditions
							/// at the free surface. Eq. 4.5
							current_cell->ftemp[i] = f_atm_eq[i] + f_atm_eq[inv] - current_cell->f[inv];
						}
						else if ((nb_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
						{
							current_cell->ftemp[i] = current_cell->f[inv];
						}
					}

					//print_fluid_amount("After interface df cycle");

					// Reconstruct atmospheric dfs for directions along the surface normal, Eq. 4.5-4.6
					/// TO DO: Check if this equation is correct. It works differently (and looks better) with a 
					/// reconstruction from an inverse fi: current_cell->ftemp[i] = f_atm_eq[i] + f_atm_eq[inv] - current_cell->f[inv]
					/// In the previous reconstruction part, finv is used because the point of reference is the nb_cell,
					/// In this case, the reference is the normal, so it should be fi...
					/// Also, in the references it only mentions that f should be updated, not ftemp; but it make more sense to 
					/// update ftemp...
					float2 normal = calculateNormal(col, row);
					for (int i = 0; i < 9; i++)
					{
						int inv = invVel2D[i];
						if (dot(normal, vel2Dv[inv]) > 0.f)
							current_cell->ftemp[i] = f_atm_eq[i] + f_atm_eq[inv] - current_cell->f[inv];
					}
				}
			}
		}
	}

	//print_fluid_amount("After stream cycle");

	//float k = 0.001f;

	//for (int row = 0; row < height; row++)
	//{
	//	for (int col = 0; col < width; col++)
	//	{
	//		d2q9_cell* current_cell = getCellAt(col, row);

	//		for (int l = 0; l < 9; l++)
	//			current_cell->f[l] = current_cell->ftemp[l];

	//		current_cell->deriveQuantities(vMax);

	//		//if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
	//		//	//current_cell->mass = current_cell->rho;
	//		//	current_cell->rho = current_cell->mass;
	//	}
	//}

	//printf("Finished stream\n\n");
}

void d2q9_lattice::collide()
{
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE && (current_cell->type & CT_EMPTY) != cellType::CT_EMPTY)
			{
				float feq[9] = { 0 };

				current_cell->deriveQuantities(vMax);
				//To include gravity, or any additional forces:
				current_cell->velocity += float2_ScalarMultiply(1.0f, float2{ 0.f, -latticeAcceleration, 0.f });
				current_cell->velocity += float2{ latticeAcceleration, 0.f, 0.f };
				//current_cell->velocity += float2{ 0.f, 0.f,0.f};

				calculateEquilibrium(current_cell->velocity, current_cell->rho, feq);

				/// Collision
				for (int i = 0; i < 9; i++)
				{
					current_cell->f[i] = (1.0f - w) * current_cell->ftemp[i] + w * feq[i];
					//current_cell->f[i] += current_cell->rho * weights2D[i] * dot(vel2Dv[i], float2{ 0.f, -latticeAcceleration, 0.f });
				}
			}
		}
	}
}

//void d2q9_lattice::updateCells()
//{
//	#pragma region // Prepare neighborhood for filled cells
//		for (float2 v : filled_cells)
//		{
//			d2q9_cell* current_cell = getCellAt(v.x, v.y);
//			for (int i = 0; i < 9; i++)
//			{
//				d2q9_cell* nb_cell = getCellAt_Mod(v.x + vel2Di[i].x, v.y + vel2Di[i].y);
//				
//				if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
//				{
//					nb_cell->type = cellType::CT_INTERFACE;
//					averageSurroundings(nb_cell, v.x + vel2Di[i].x, v.y + vel2Di[i].y);
//				}
//
//				if (nb_cell->type & )
//				emptied_cells
//			}
//		}
//	#pragma endregion
//}

void d2q9_lattice::updateCells()
{
	float k = 0.001f, lonely_thresh = 0.1f;

#pragma region	// check whether interface cells emptied or filled
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < height; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
			{
				// Interfase cell filled
				if (
					(current_cell->mass >((1.f + k) * current_cell->rho))
					||
					(
					(current_cell->mass >= (1 - lonely_thresh) * current_cell->rho) &&
					//((current_cell->type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
					((current_cell->type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
					)
					)
				{
					current_cell->type |= cellType::CT_IF_TO_FLUID;
				}
				else if (
					(current_cell->mass < ((- k) * current_cell->rho))
					||
					(
					(current_cell->mass <= lonely_thresh * current_cell->rho) &&
					((current_cell->type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
					|| ((current_cell->type & CT_NO_IFACE_NEIGH) == CT_NO_IFACE_NEIGH &&
						(current_cell->type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH
						)
					)
					)
				{
					current_cell->type |= cellType::CT_IF_TO_EMPTY;
				}
			}

			current_cell->type &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
		}
	}
#pragma endregion

#pragma region	// set flags for filled interface cells (interface to fluid)
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			// Initialize values for empty cells
			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
			{
				for (int i = 0; i < 9; i++)
				{
					d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[i].x, row + vel2Di[i].y);

					if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
					{
						nb_cell->type = cellType::CT_INTERFACE;
						averageSurroundings(nb_cell, col + vel2Di[i].x, row + vel2Di[i].y);
					}

					//// prevent neighboring cells from becoming empty
					if ((nb_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
						nb_cell->type = cellType::CT_INTERFACE;
				}

				//// prevent neighboring cells from becoming empty
				//for (int i = 1; i < 9; i++)
				//{
				//	d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[i].x, row + vel2Di[i].y);

				//	if ((nb_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY || (nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
				//		nb_cell->type = cellType::CT_INTERFACE;
				//}
			}
		}
	}
#pragma endregion

#pragma region	// set flags for emptied interface cells (interface to empty)
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			// convert neighboring fluid cells to interface cells
			if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
			{
				for (int i = 0; i < 9; i++)
				{
					d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[i].x, row + vel2Di[i].y);

					if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
						nb_cell->type = cellType::CT_INTERFACE;
				}
			}
		}
	}
#pragma endregion

	/// In a second pass, the excess mass mex is distributed among the surrounding interface cells for each emptied and filled cell.
#pragma region // distribute excess mass
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);
			//float2 normal = calculateNormal(col, row);
			float excess_mass = 0.f;

			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
			{
				excess_mass = current_cell->mass - current_cell->rho;
				//if (current_cell->mass > 1.0)
				//	int test = 0;
				//current_cell->mass = 1.0f;// current_cell->rho;
			}
			else if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
			{
				excess_mass = current_cell->mass;
				//if (excess_mass > 0.f)
				//	int test = 0;
				//normal = float2_ScalarMultiply(-1.f, normal);
				//current_cell->mass = 0.f;
			}
			else
			{
				continue;
			}

			float eta[9] = { 0.f };
			float eta_total = 0.f;

			for (int i = 0; i < 9; i++)
			{
				// neighbor cell in the direction of velocity vector
				d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[invVel2D[i]].x, row + vel2Di[invVel2D[i]].y);
				if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				{
					float2 normal = calculateNormal(col + vel2Di[invVel2D[i]].x, row + vel2Di[invVel2D[i]].y);
					if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
					{
						eta[i] = dot(vel2Di[i], normal);
						if (eta[i] <= 0) eta[i] = 0;
					}

					if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
					{
						eta[i] = dot(vel2Di[i], normal);
						if (eta[i] < 0)
						{
							normal = float2_ScalarMultiply(-1.f, normal);
							eta[i] = dot(vel2Di[i], normal);
						}
						else
							eta[i] = 0.f;
					}

					eta_total += eta[i];
				}
			}

			/// TODO
			// store excess mass to be distributed in mex
			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID ||
				(current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
				if (eta_total > 1.e-6 || eta_total < -1.e-6)
				{
					float eta_fraction = 1.f / eta_total;
					for (int i = 0; i < 9; i++)
					{
						float excess = excess_mass * eta[i] * eta_fraction;

						if (excess > 1.e-6)
							current_cell->mex[i] = excess;
						/*else if (excess < -1.e-6)
							current_cell->mex[i] = excess;*/
						else
							current_cell->mex[i] = 0.f;
					}

				}
		}
	}
#pragma endregion

#pragma region // collect distributed mass and finalize cell flags
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID ||
				(current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
			{
				for (int i = 0; i < 9; i++)
				{
					d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[invVel2D[i]].x, row + vel2Di[invVel2D[i]].y);

					if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
					{
						nb_cell->mass += current_cell->mex[i];
						if (nb_cell->mass > nb_cell->rho)
							nb_cell->mass = nb_cell->rho;
						current_cell->mex[i] = 0.f;
					}
				}
			}

			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
			{
				current_cell->type = cellType::CT_FLUID;
				current_cell->mass = current_cell->rho = 1.0f;
			}
			else if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
			{
				current_cell->type = cellType::CT_EMPTY;
				current_cell->mass = 0.f;
			}
		}
	}
#pragma endregion

#pragma region // set neighborhood flags
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);

			if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE)
			{
				current_cell->type |= (cellType::CT_NO_FLUID_NEIGH | cellType::CT_NO_IFACE_NEIGH | cellType::CT_NO_EMPTY_NEIGH);

				for (int i = 0; i < 9; i++)
				{
					d2q9_cell* nb_cell = getCellAt_Mod(col + vel2Di[i].x, row + vel2Di[i].y);

					if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
						current_cell->type &= ~cellType::CT_NO_FLUID_NEIGH;
					if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
						current_cell->type &= ~cellType::CT_NO_IFACE_NEIGH;
					if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
						current_cell->type &= ~cellType::CT_NO_EMPTY_NEIGH;
				}

				//if ((current_cell->type & CT_NO_EMPTY_NEIGH) == cellType::CT_NO_EMPTY_NEIGH)
				//	current_cell->type &= ~cellType::CT_NO_FLUID_NEIGH;
			}
		}
	}
#pragma endregion
}

void d2q9_lattice::averageSurroundings(d2q9_cell* cell_, int x_, int y_)
{
	int counter = 0;
	cell_->mass = 0.f;
	cell_->rho = 0.f;
	cell_->velocity = float2{ 0.f, 0.f };

	for (int i = 0; i < 9; i++)
	{
		d2q9_cell* nb_cell = getCellAt_Mod(x_ + vel2Di[invVel2D[i]].x, y_ + vel2Di[invVel2D[i]].y);

		if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID || (nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
		{
			cell_->rho += nb_cell->rho;
			cell_->velocity += nb_cell->velocity;
			counter++;
		}
	}
	if (counter > 0)
	{
		cell_->rho /= (float)counter;
		cell_->velocity /= (float)counter;
	}

	calculateEquilibrium(cell_->velocity, cell_->rho, cell_->f);
}

void d2q9_lattice::calculateEquilibrium(float2 velocity_, float rho_, float feq_[9])
{
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(velocity_, velocity_);	//Velocity squared

	for (int i = 0; i < 9; i++)
	{
		eiU = dot(vel2Dv[i], velocity_);
		feq_[i] = weights2D[i] * (rho_ + 3.f * eiU - 1.5f * uSq + 4.5f * eiU * eiU);
	}
}

void d2q9_lattice::initCells(int **typeArray_, float initRho_, float2 initVelocity_)
{
	float feq[9] = { 0 };
	d2q9_cell *current_cell;

	for (int row = 0; row < height; row++)
	for (int col = 0; col < width; col++)
	{
		current_cell = getCellAt(col, row);
		current_cell->type = typeArray_[row][col];

		if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID || (current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
		{
			for (int l = 0; l < 9; l++)
				current_cell->f[l] = current_cell->ftemp[l] = weights2D[l];
			
			if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
				current_cell->mass = 1.0f;

			// (arbitrarily) assign very little mass
			if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				current_cell->mass = 0.1f;

			current_cell->deriveQuantities(vMax);
		}
	}

	/// Stores the initial mass for each type of cell; for debugging purposes.
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			d2q9_cell* current_cell = getCellAt(col, row);
			if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
			{
				initial_fluid_mass += current_cell->mass;
				initial_fluid_cells++;
			}
			else if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
			{
				initial_interface_mass += current_cell->mass;
				initial_interface_cells++;
			}
			else if ((current_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
			{
				initial_air_mass += current_cell->mass;
				initial_air_cells++;
			}
			else if ((current_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
			{
				initial_solid_mass += current_cell->mass;
			}
		}
	}
	initial_mas = initial_fluid_mass + initial_interface_mass + initial_air_mass + initial_solid_mass;

	printf("\n Initial mass %f\n f mass %f\n i mass %f\n a mass %f \n s mass %f \n\n", initial_mas, initial_fluid_mass, initial_interface_mass,
		initial_air_mass, initial_solid_mass);
	printf("\n Initial cells. F %d I %d A %d\n", initial_fluid_cells, initial_interface_cells, initial_air_cells);
	printf("Initial viscosity %f\n\n", w);
}

float2 d2q9_lattice::calculateNormal(int x_, int y_)
{
	//float2 normal;
	//normal.x = 0.5f * (getCellAt_Mod(x_ - 1, y_)->calculateEpsilon() - getCellAt_Mod(x_ + 1, y_)->calculateEpsilon());
	//normal.y = 0.5f * (getCellAt_Mod(x_, y_ - 1)->calculateEpsilon() - getCellAt_Mod(x_, y_ + 1)->calculateEpsilon());
	//return normal;

	float2 normal;
	normal.x = 0.5f * (getCellAt_Mod(x_ - 1, y_)->epsilon - getCellAt_Mod(x_ + 1, y_)->epsilon);
	normal.y = 0.5f * (getCellAt_Mod(x_, y_ - 1)->epsilon - getCellAt_Mod(x_, y_ + 1)->epsilon);
	return normal;
}

float d2q9_lattice::calculateMassExchange(int current_type, int nb_type, float fi_neigh, float fi_inv)
{
	if ((current_type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
	{
		if ((nb_type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
			return fi_neigh - fi_inv;
		else
			return -fi_inv;
	}
	else if ((current_type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
	{
		if ((nb_type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			return fi_neigh - fi_inv;
		else
			return fi_neigh;
	}
	else /// Current cell is standard cell
	{
		if ((nb_type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
			return fi_neigh;
		else if ((nb_type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			return -fi_inv;
		else
			return fi_neigh - fi_inv;
	}
	//return 0.f;
}