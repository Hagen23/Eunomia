// Sergio's
#include "Lattice_3D.h"
#include <memory>

d3q19_cell::d3q19_cell() : mass(0.0f), type(cellType::CT_EMPTY), rho(0.0f), velocity(float3{ 0.0f, 0.0f, 0.0f })
{
	for (int i = 0; i < 19; i++)
		f[i] = ftemp[i] = mex[i] = 0.f;
}

d3q19_cell::d3q19_cell(float *f_, float mass_ = 0.0f, float rho_ = 0.0f,
	int type_ = cellType::CT_EMPTY, float3 velocity_ = float3{ 0.0f, 0.0f, 0.0f })
{
	for (int i = 0; i < 19; i++)
	{
		f[i] = ftemp[i] = f_[i];
		mex[i] = 0.f;
	}

	type = type_;
	rho = rho_;
	mass = mass_;
	velocity = velocity_;
}

void d3q19_cell::deriveQuantities(float vMax_)
{
	rho = 0;
	velocity = float3{ 0.0f, 0.0f, 0.0f };

	for (int i = 0; i < 19; i++)
		rho += f[i];

	/// rho * v = sum(fi * ei)
	if (rho > 0.0f)
	{
		for (int i = 0; i < 19; i++)
			velocity += float3_ScalarMultiply(f[i], vel3Dv[i]);

		velocity = float3_ScalarMultiply(1.0f / rho, velocity);
	}

	/// Rescale in case maximum velocity is exceeded
	float norm = float3_Norm(velocity);
	if (norm > vMax_)
		velocity = float3_ScalarMultiply(vMax_ / norm, velocity);
}

float d3q19_cell::calculateEpsilon()
{
	//if ((type & CT_FLUID) == cellType::CT_FLUID || (type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
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
		//	return 0.1f;
		//}
	}

	return 0.f;
}

d3q19_lattice::d3q19_lattice(int width_, int height_, int depth_, float worldViscosity_, float cellsPerSide_, float domainSize_)
{
	width = width_;
	height = height_;
	depth = depth_;
	stride = 19;

	// The cells are initiated statically with the predefined size. This initialization step would be needed with dynamic pointer
	cells = new d3q19_cell[width*height*depth]();

	// Values for stability and to integrate gravity
	c = (float)(1.0 / sqrt(3.0));
	domainSize = domainSize_;
	cellsPerSide = cellsPerSide_;
	cellSize = domainSize_ / cellsPerSide_;
	gravity = 9.81f;

	/// Stability concerns, section 3.3 of the reference thesis.
	/// Here a value of gc = 0.005 is used to keep the compressibility below half a percent.
	timeStep = (float)(sqrtf((0.005f * cellSize) / fabs(gravity)));

	vMax = cellSize / timeStep;
	viscosity = worldViscosity_ * timeStep / (cellSize * cellSize);
	tau = 3.0f * viscosity + 0.5f;
	w = 1.0f / tau;
	latticeAcceleration = gravity * timeStep * timeStep / cellSize;

	minMassExchange = 10;
	exchange_counter = 0;
}

void d3q19_lattice::step()
{
	//collide();
	//stream();
	//updateCells();
	//print_fluid_amount();

	print_fluid_amount("Before stream");
	stream();
	print_fluid_amount("After stream");
	collide(); 
	print_fluid_amount("After collide");
	updateCells();
	print_fluid_amount("After update");
	printf("\n\n");

	//print_types();
}

void d3q19_lattice::stream()
{
	for (int slice = 0; slice< depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				//print_fluid_amount("Before df cycle");

				if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE && (current_cell->type & CT_EMPTY) != cellType::CT_EMPTY)
				{
					if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
					{
						for (int i = 0; i < 19; i++)
						{
							// f'i(x, t+dt) = fi(x+e�, t)
							d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

							/// Normal streaming, no mass exchange is needed -- or is it?
							if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)// || (nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
							{
								//float test = nb_cell->f[i] - current_cell->f[invVel3D[i]];

								//if (abs(test) > 0.000001)
								//	current_cell->mass += nb_cell->f[i] - current_cell->f[invVel3D[i]];

								current_cell->ftemp[i] = nb_cell->f[i];
							}
							else if ((nb_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
							{
								current_cell->ftemp[i] = current_cell->f[invVel3D[i]];
							}
						}
						//print_fluid_amount("After fluid df cycle");
					}
					else if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
					{
						//print_fluid_amount("Before interface df cycle");
						const float current_epsilon = current_cell->calculateEpsilon();
						float f_atm_eq[19] = { 0 };

						/// Equilibrium calculated with air density, and the velocity of the current cell. Used to 
						/// reconstruct missing dfs.
						calculateEquilibrium(current_cell->velocity, 1.f, f_atm_eq);

						for (int i = 0; i < 19; i++)
						{
							// f'i(x, t+dt) = fi(x+e�, t)
							d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

							if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
							{
								// dmi(x, t + dt) = fi(x+e�,t) - f�(x, t)
								// fi(x+e�,t) -> mass incoming from fluid;  f�(x, t) -> mass outgoing from interfase
								//current_cell->mass += nb_cell->f[i] - current_cell->f[invVel3D[i]];

								float test = (nb_cell->f[i] - current_cell->f[invVel3D[i]]);
								
								if (abs(test) > 0.000001)
								{
									current_cell->mass += nb_cell->f[i] - current_cell->f[invVel3D[i]];
								}

								/// Normal streaming
								current_cell->ftemp[i] = nb_cell->f[i];
							}
							else if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
							{
								const float nb_epsilon = nb_cell->calculateEpsilon();

								/// Eq. 4.3
								//current_cell->mass += (nb_cell->f[i] - current_cell->f[invVel3D[i]]) * 0.5f * (current_epsilon + nb_epsilon);
								//current_cell->mass +=
									//calculateMassExchange(current_cell->type, nb_cell->type, nb_cell->f[i], current_cell->f[invVel3D[i]]) *
									//0.5f * (current_epsilon + nb_epsilon);

								float test = calculateMassExchange(current_cell->type, nb_cell->type, nb_cell->f[i], current_cell->f[invVel3D[i]]) *
									0.5f * (current_epsilon + nb_epsilon);

								if (abs(test) > 0.000001)
								{
									current_cell->mass +=
										calculateMassExchange(current_cell->type, nb_cell->type, nb_cell->f[i], current_cell->f[invVel3D[i]]) *
										0.5f * (current_epsilon + nb_epsilon);
								}
								current_cell->ftemp[i] = nb_cell->f[i];
							}
							else if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
							{
								/// no mass exchange from or to empty cell
								/// DFs that would come out of the empty cells need to be reconstructed from the boundary conditions
								/// at the free surface. Eq. 4.5
								current_cell->ftemp[i] = f_atm_eq[i] + f_atm_eq[invVel3D[i]] - current_cell->f[invVel3D[i]];
							}
							else if ((nb_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
							{
								current_cell->ftemp[i] = current_cell->f[invVel3D[i]];
							}
						}

						//print_fluid_amount("After interface df cycle");

						// Reconstruct atmospheric dfs for directions along the surface normal
						float3 normal = calculateNormal(col, row, slice);
						for (int i = 0; i < 19; i++)
						{
							if (dot(normal, vel3Dv[invVel3D[i]]) > 0.f)
								current_cell->ftemp[i] = f_atm_eq[i] + f_atm_eq[invVel3D[i]] - current_cell->f[invVel3D[i]];
						}
					}
				}
			}
		}
	}

	//print_fluid_amount("After stream cycle");

	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				for (int l = 0; l < 19; l++)
					current_cell->f[l] = current_cell->ftemp[l];

				current_cell->deriveQuantities(vMax);

				//if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
				//	//current_cell->mass = current_cell->rho;
				//	current_cell->rho = current_cell->mass;
			}
		}
	}

	//printf("Finished stream\n\n");
}

void d3q19_lattice::collide()
{
	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE && (current_cell->type & CT_EMPTY) != cellType::CT_EMPTY)
				{
					float feq[19] = { 0 };
					current_cell->velocity += float3{ 0.f, -latticeAcceleration, 0.f };
					//current_cell->velocity += float3{ 0.f, -latticeAcceleration * tau / current_cell->rho, 0.f };
					calculateEquilibrium(current_cell->velocity, current_cell->rho, feq);

					/// Collision
					for (int i = 0; i < 19; i++)
					{
						current_cell->f[i] = (1.0f - w) * current_cell->f[i] + w * feq[i];
						//To include gravity
						//current_cell->f[i] += current_cell->rho * weights3D[i] * dot(vel3Dv[i], float3{ 0.f, -latticeAcceleration, 0.f });
					}
				}
			}
		}
	}
}

void d3q19_lattice::updateCells()
{
	float k = 0.001f, lonely_thresh = 0.1f;

#pragma region	// check whether interface cells emptied or filled
	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < height; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				{
					// Interfase cell filled
					if ((current_cell->mass >((1.f + k) * current_cell->rho)) ||
						((current_cell->mass >= (1 - lonely_thresh) * current_cell->rho) &&
						(current_cell->type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH))
					{
						filled_cells.push_back(int3{ col, row, slice });
						current_cell->type = cellType::CT_IF_TO_FLUID;
					}
					else if ((current_cell->mass < ((0.f - k) * current_cell->rho)) ||
						(current_cell->mass <= lonely_thresh * current_cell->rho) &&
						((current_cell->type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH) ||
						((current_cell->type & CT_NO_IFACE_NEIGH) == CT_NO_IFACE_NEIGH && (current_cell->type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
						)
					{
						emptied_cells.push_back(int3{ col, row, slice });
						current_cell->type = cellType::CT_IF_TO_EMPTY;
					}
				}

				current_cell->type &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
			}
		}
	}
#pragma endregion
	print_fluid_amount("After checking for filled or emptied");

#pragma region	// set flags for filled interface cells (interface to fluid)
	for (int3 current_position : filled_cells)
	{
		d3q19_cell* current_cell = getCellAt(current_position);

		for (int i = 0; i < 19; i++)
		{
			d3q19_cell* nb_cell = getCellAt_Mod(current_position.x - vel3Di[i].x, 
				current_position.y - vel3Di[i].y, current_position.z -vel3Di[i].z);

			if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
			{
				nb_cell->type = cellType::CT_INTERFACE;
				//averageSurroundings(nb_cell, col, row);
				averageSurroundings(nb_cell, current_position.x - vel3Di[i].x,
					current_position.y - vel3Di[i].y, current_position.z - vel3Di[i].z);
			}
		}

		// prevent neighboring cells from becoming empty
		for (int i = 0; i < 19; i++)
		{
			d3q19_cell* nb_cell = getCellAt_Mod(current_position.x - vel3Di[i].x,
				current_position.y - vel3Di[i].y, current_position.z - vel3Di[i].z);

			if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
			{
				vector<int3>::iterator element = find(emptied_cells.begin(), emptied_cells.end(), int3{ current_position.x - vel3Di[i].x,
					current_position.y - vel3Di[i].y, current_position.z - vel3Di[i].z });
				
				if (element != emptied_cells.end())
					emptied_cells.erase(element);
			}
		}

		current_cell->type = CT_FLUID;
	}
	//for (int slice = 0; slice < depth; slice++)
	//{
	//	for (int row = 0; row < height; row++)
	//	{
	//		for (int col = 0; col < width; col++)
	//		{
	//			d3q19_cell* current_cell = getCellAt(col, row, slice);

	//			// Initialize values for empty cells
	//			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
	//			{
	//				for (int i = 0; i < 19; i++)
	//				{
	//					d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

	//					if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
	//					{
	//						nb_cell->type = cellType::CT_INTERFACE;
	//						//averageSurroundings(nb_cell, col, row);
	//						averageSurroundings(nb_cell, col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);
	//					}
	//				}

	//				// prevent neighboring cells from becoming empty
	//				for (int i = 0; i < 19; i++)
	//				{
	//					d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

	//					if ((nb_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY || (nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
	//						nb_cell->type = cellType::CT_INTERFACE;
	//				}
	//			}
	//		}
	//	}
	//}
#pragma endregion
	print_fluid_amount("After flags for interface to fluid");

#pragma region	// set flags for emptied interface cells (interface to empty)
	for (int3 current_position : emptied_cells)
	{
		d3q19_cell* current_cell = getCellAt(current_position);

		// convert neighboring fluid cells to interface cells
		for (int i = 0; i < 19; i++)
		{
			d3q19_cell* nb_cell = getCellAt_Mod(current_position.x - vel3Di[i].x,
				current_position.y - vel3Di[i].y, current_position.z - vel3Di[i].z);

			if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
				nb_cell->type = cellType::CT_INTERFACE;
		}

		current_cell->type = CT_EMPTY;
	}
	//for (int slice = 0; slice < depth; slice++)
	//{
	//	for (int row = 0; row < height; row++)
	//	{
	//		for (int col = 0; col < width; col++)
	//		{
	//			d3q19_cell* current_cell = getCellAt(col, row, slice);

	//			// convert neighboring fluid cells to interface cells
	//			if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
	//			{
	//				for (int i = 0; i < 19; i++)
	//				{
	//					d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

	//					if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
	//						nb_cell->type = cellType::CT_INTERFACE;
	//				}
	//			}
	//		}
	//	}
	//}
#pragma endregion

	print_fluid_amount("After flags for interface to empty");
	/// In a second pass, the excess mass mex is distributed among the surrounding interface cells for each emptied and filled cell.
#pragma region // distribute excess mass
	for (int3 current_position : filled_cells)
	{
		float excess_mass = 0.f, excess = 0.f;

		d3q19_cell* current_cell = getCellAt(current_position);

		excess = current_cell->mass - current_cell->rho;

		// Fluid moved beyond the current cell during the last time step
		if ((excess > 0.f) && (excess > current_cell->rho))
		{
			float3 normal = calculateNormal(current_position.x, current_position.y, current_position.z);

			float eta[19] = { 0.f };
			float eta_total = 0.f;

			for (int i = 0; i < 19; i++)
			{
				float nei = dot(normal, vel3Dv[i]);
				eta[i] = nei > 0.f ? nei : 0.f;
				eta_total += eta[i];
			}
		}
	}
	//for (int slice = 0; slice < depth; slice++)
	//{
	//	for (int row = 0; row < height; row++)
	//	{
	//		for (int col = 0; col < width; col++)
	//		{
	//			d3q19_cell* current_cell = getCellAt(col, row, slice);
	//			float3 normal = calculateNormal(col, row, slice);
	//			float excess_mass = 0.f;

	//			if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
	//			{
	//				float excess = current_cell->mass - current_cell->rho;
	//				
	//				if (abs(excess) > 0.000001f)
	//					excess = 0.000001f;

	//				excess_mass = excess; // current_cell->mass - current_cell->rho;
	//				current_cell->mass = current_cell->rho;
	//				//current_cell->mass = 1.f;
	//			}
	//			else if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
	//			{
	//				excess_mass = current_cell->mass;
	//				normal = float3_ScalarMultiply(-1.f, normal);
	//				current_cell->mass = 0.f;
	//			}
	//			else
	//			{
	//				continue;
	//			}

	//			float eta[19] = { 0.f };
	//			float eta_total = 0.f;
	//			unsigned int isIF[19] = { 0 };
	//			unsigned int numIF = 0;	// number of interface cell neighbors

	//			for (int i = 0; i < 19; i++)
	//			{
	//				// neighbor cell in the direction of velocity vector
	//				d3q19_cell* nb_cell = getCellAt_Mod(col + vel3Di[i].x, row + vel3Di[i].y, slice + vel3Di[i].z);
	//				if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
	//				{
	//					eta[i] = dot(vel3Di[i], normal);
	//					if (eta[i] < 0) eta[i] = 0;

	//					eta_total += eta[i];
	//					isIF[i] = 1;
	//					numIF++;
	//				}
	//			}

	//			// store excess mass to be distributed in mex
	//			if (eta_total > 0)
	//			{
	//				float eta_fraction = 1 / eta_total;
	//				for (int i = 0; i < 19; i++)
	//					current_cell->mex[i] = excess_mass * eta[i] * eta_fraction;
	//			}
	//			//else if (numIF > 0)
	//			//{
	//			//	float excess_mass_uniform = excess_mass / numIF;
	//			//	for (int i = 0; i < 9; i++)
	//			//		current_cell->mex[i] = isIF[i] ? excess_mass_uniform : 0.f;
	//			//}
	//		}
	//	}
	//}
#pragma endregion

	print_fluid_amount("After excess mass calculation");

#pragma region // collect distributed mass and finalize cell flags
	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID ||
					(current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
				{
					for (int i = 0; i < 19; i++)
					{
						d3q19_cell* nb_cell = getCellAt_Mod(col + vel3Di[i].x, row + vel3Di[i].y, slice + vel3Di[i].z);

						if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
						{
							if (nb_cell->mass < 1.f)
								nb_cell->mass += current_cell->mex[i];
							current_cell->mex[i] = 0.f;
						}
					}
				}

				if ((current_cell->type & CT_IF_TO_FLUID) == cellType::CT_IF_TO_FLUID)
					current_cell->type = cellType::CT_FLUID;
				else if ((current_cell->type & CT_IF_TO_EMPTY) == cellType::CT_IF_TO_EMPTY)
					current_cell->type = cellType::CT_EMPTY;
			}
		}
	}
#pragma endregion

	print_fluid_amount("After excess mass distribution");

#pragma region // set neighborhood flags
	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE)
				{
					current_cell->type |= (cellType::CT_NO_FLUID_NEIGH | cellType::CT_NO_IFACE_NEIGH | cellType::CT_NO_EMPTY_NEIGH);

					for (int i = 0; i < 19; i++)
					{
						d3q19_cell* nb_cell = getCellAt_Mod(col - vel3Di[i].x, row - vel3Di[i].y, slice - vel3Di[i].z);

						if ((nb_cell->type & CT_FLUID) == cellType::CT_FLUID)
							current_cell->type &= ~cellType::CT_NO_FLUID_NEIGH;
						if ((nb_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
							current_cell->type &= ~cellType::CT_NO_IFACE_NEIGH;
						if ((nb_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
							current_cell->type &= ~cellType::CT_NO_EMPTY_NEIGH;
					}

					if ((current_cell->type & CT_NO_EMPTY_NEIGH) == cellType::CT_NO_EMPTY_NEIGH)
						current_cell->type &= ~cellType::CT_NO_FLUID_NEIGH;
				}
			}
		}
	}
#pragma endregion
}

void d3q19_lattice::averageSurroundings(d3q19_cell* cell_, int x_, int y_, int z_)
{
	int counter = 0;
	cell_->mass = 0.f;
	cell_->rho = 0.f;
	cell_->velocity = float3{ 0.f, 0.f, 0.f };

	for (int i = 0; i < 19; i++)
	{
		d3q19_cell* nb_cell = getCellAt_Mod(x_ - vel3Di[i].x, y_ - vel3Di[i].y, z_ - vel3Di[i].z);

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

void d3q19_lattice::calculateEquilibrium(float3 velocity_, float rho_, float feq_[19])
{
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(velocity_, velocity_);	//Velocity squared

	for (int i = 0; i < 19; i++)
	{
		eiU = dot(vel3Dv[i], velocity_);
		//feq_[i] = weights3D[i] * (rho_ + 3.f * eiU - 1.5f * uSq + 4.5f * eiU * eiU);
		feq_[i] = weights3D[i] * rho_ * (1.f + 3.f * eiU - 1.5f * uSq + 4.5f * eiU * eiU);
	}
}

void d3q19_lattice::initCells(int *typeArray_, float initRho_, float3 initVelocity_)
{
	float feq[19] = { 0 };
	d3q19_cell *current_cell;
	calculateEquilibrium(initVelocity_, initRho_, feq);

	for (int slice = 0; slice < depth; slice++)
	for (int row = 0; row < height; row++)
	for (int col = 0; col < width; col++)
	{
		current_cell = getCellAt(col, row, slice);
		current_cell->type = typeArray_[I3D(width, height, col, row, slice)];

		if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID || (current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
		{
			for (int l = 0; l < 19; l++)
				current_cell->f[l] = current_cell->ftemp[l] = feq[l];

			current_cell->deriveQuantities(vMax);

			if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
				//current_cell->mass = current_cell->rho;
				current_cell->mass = 1.f;

			// (arbitrarily) assign very little mass
			if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				//current_cell->mass = 1.f / 5.f;//0.1f * current_cell->rho;
				current_cell->mass = 1.f;
		}
	}

	/// Stores the initial mass for each type of cell
	for (int slice = 0; slice < depth; slice++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				d3q19_cell* current_cell = getCellAt(col, row, slice);

				if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
				{
					initial_fluid_mass += current_cell->mass;
				}
				else if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
				{
					initial_interface_mass += current_cell->mass;
				}
				else if ((current_cell->type & CT_EMPTY) == cellType::CT_EMPTY)
				{
					initial_air_mass += current_cell->mass;
				}
				else if ((current_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
				{
					initial_solid_mass += current_cell->mass;
				}
			}
		}
	}
	initial_mas = initial_fluid_mass + initial_interface_mass + initial_air_mass + initial_solid_mass;

	printf("\n Viscosity %f\n", w);
	printf("\n Initial mass %f\n f mass %f\n i mass %f\n a mass %f \n s mass %f \n\n", initial_mas, initial_fluid_mass, initial_interface_mass,
		initial_air_mass, initial_solid_mass);
}

float3 d3q19_lattice::calculateNormal(int x_, int y_, int z_)
{
	float3 normal;
	normal.x = 0.5f * (getCellAt_Mod(x_ - 1, y_, z_)->calculateEpsilon() - getCellAt_Mod(x_ + 1, y_, z_)->calculateEpsilon());
	normal.y = 0.5f * (getCellAt_Mod(x_, y_ - 1, z_)->calculateEpsilon() - getCellAt_Mod(x_, y_ + 1, z_)->calculateEpsilon());
	normal.z = 0.5f * (getCellAt_Mod(x_, y_, z_ - 1)->calculateEpsilon() - getCellAt_Mod(x_, y_, z_ + 1)->calculateEpsilon());
	return normal;
}

float d3q19_lattice::calculateMassExchange(int current_type, int nb_type, float fi_neigh, float fi_inv)
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
		if ((current_type & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
			return fi_neigh;
		else if ((nb_type & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			return -fi_inv;
		else
			return fi_neigh - fi_inv;
	}
	return 0.f;
}

// ////MY version
//#include "Lattice_2D.h"
//#include <memory>
//
//d3q19_cell::d3q19_cell() : mass(0.0f), type(cellType::CT_EMPTY), rho(0.0f), velocity(float3{ 0.0f, 0.0f }), mass_temp(0.0f)
//{
//	//f = new float[9]();
//	//ftemp = new float[9]();
//	for (int i = 0; i < 9; i++)
//		f[i] = ftemp[i] = mex[i] = 0.f;
//}
//
//d3q19_cell::d3q19_cell(float *f_, float mass_ = 0.0f, float rho_ = 0.0f,
//	int type_ = cellType::CT_EMPTY, float3 velocity_ = float3{ 0.0f, 0.0f })
//{
//	for (int i = 0; i < 9; i++)
//	{
//		f[i] = ftemp[i] = f_[i];
//		mex[i] = 0.f;
//	}	
//	//f = new float[9]();
//	//ftemp = new float[9]();
//	//memcpy(&f, f_, sizeof(float)* 9);
//	type = type_;
//	rho = rho_;
//	mass = mass_;
//	mass_temp = 0.f;
//	velocity = velocity_;
//}
//
//void d3q19_cell::deriveQuantities(float vMax_)
//{
//	rho = 0;
//	velocity = float3{ 0.0f, 0.0f };
//
//	for (int i = 0; i < 9; i++)
//		rho += f[i];
//		
//	if (rho > 0.0f)
//	{
//		for (int i = 0; i < 9; i++)
//			velocity += float3_ScalarMultiply(f[i], vel3Dv[i]);
//		
//		velocity = float3_ScalarMultiply(1.0f / rho, velocity);
//	}
//
//	float norm = float3_Norm(velocity);
//	if (norm > vMax_)
//		velocity = float3_ScalarMultiply(vMax_ / norm, velocity);
//}
//
//float d3q19_cell::calculateEpsilon()
//{
//	if (type == cellType::CT_FLUID || type == cellType::CT_OBSTACLE)
//		return 1.f;
//
//	if (type == cellType::CT_EMPTY)
//		return 0.f;
//
//	if (type == cellType::CT_INTERFACE)
//	{
//		if (rho > 0)
//		{
//			float epsilon = mass / rho;
//			// df->mass can even be < 0 or > df->rho for interface cells to be converted to fluid or empty cells in the next step;
//			// clamp to [0,1] range for numerical stability
//			if (epsilon > 1)
//				return 1.f;
//			else if (epsilon < 0)
//				return 0.f;
//			else
//				return epsilon;
//		}
//		else
//		{
//			// return (somewhat arbitrarily) a ratio of 1/2 
//			return 0.5f;
//		}
//	}
//	
//	return 0;
//}
//
//d3q19_lattice::d3q19_lattice(int width_, int height_, float worldViscosity_, float cellsPerSide_, float domainSize_)
//{
//	width = width_;
//	height = height_;
//	stride = 9;
//	
//	// The cells are initiated statically with the predefined size. This initialization step would be needed with dynamic pointer
//	//cells = new d3q19_cell*[width];
//	//for (int i = 0; i < width; i++)
//	//	cells[i] = new d3q19_cell[height];
//
//	// Values for stability and to integrate gravity
//	c = (float)(1.0 / sqrt(3.0));
//	domainSize = domainSize_;
//	cellsPerSide = cellsPerSide_;
//	cellSize = domainSize_ / cellsPerSide_;
//	gravity = 9.81f;
//	timeStep = (float)(sqrtf((0.005f * cellSize) / fabs(gravity)));
//	vMax = cellSize / timeStep;
//	viscosity = worldViscosity_ * timeStep / (cellSize * cellSize);
//	tau = 3.0f * viscosity + 0.5f;
//	w = 1.0f / tau;
//	latticeAcceleration = gravity * timeStep * timeStep / cellSize;
//}
//
//void d3q19_lattice::step()
//{
//	stream();
//	collide();
//	updateCells();
//	print_fluid_amount();
//}
//
//void d3q19_lattice::stream()
//{
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			if (current_cell->type != cellType::CT_OBSTACLE && current_cell->type != cellType::CT_EMPTY)
//			{
//				if (current_cell->type == cellType::CT_FLUID)
//				{
//					for (int l = 0; l < 9; l++)
//					{
//						// f'i(x, t+dt) = fi(x+e�, t)
//						d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//
//						if (nb_cell->type == cellType::CT_FLUID || nb_cell->type == cellType::CT_INTERFACE)
//						{
//							current_cell->mass_temp += nb_cell->f[l] - current_cell->f[invVel3D[l]];
//							current_cell->ftemp[l] = nb_cell->f[l];
//						}
//						else if (nb_cell->type == cellType::CT_OBSTACLE)
//						{
//							current_cell->ftemp[l] = current_cell->f[invVel3D[l]];
//						}
//					}
//				}
//				else if (current_cell->type == cellType::CT_INTERFACE)
//				{
//					float f_atm_eq[9] = { 0 };
//					calculateEquilibrium(current_cell->velocity, 1.f, f_atm_eq);
//
//					for (int l = 0; l < 9; l++)
//					{
//						// f'i(x, t+dt) = fi(x+e�, t)
//						d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//
//						if (nb_cell->type == cellType::CT_FLUID)
//						{
//							// dmi(x, t + dt) = fi(x+e�,t) - f�(x, t)
//							// fi(x+e�,t) -> mass incoming from fluid;  f�(x, t) -> mass outgoing from interfase
//							current_cell->mass_temp += nb_cell->f[l] - current_cell->f[invVel3D[l]];
//							
//							current_cell->ftemp[l] = nb_cell->f[l];
//						}
//						else if (nb_cell->type == cellType::CT_INTERFACE)
//						{
//							const float current_epsilon = current_cell->calculateEpsilon();
//							const float nb_epsilon = nb_cell->calculateEpsilon();
//
//							current_cell->mass_temp += (nb_cell->f[l] - current_cell->f[invVel3D[l]])*
//								0.5f * (current_epsilon + nb_epsilon);
//							//current_cell->mass_temp += (nb_cell->f[invVel3D[l]] - current_cell->f[l])*0.5f * (current_epsilon + nb_epsilon);
//
//							// TO DO: Integrate the table values to remove visual artifacts
//
//							current_cell->ftemp[l] = nb_cell->f[l];
//						}
//						else if (nb_cell->type == cellType::CT_EMPTY)
//						{
//							// no mass exchange from or to empty cell
//							// DFs that would come out of the empty cells need to be reconstructed from the boundary conditions at the free surface
//							current_cell->ftemp[l] = f_atm_eq[l] + f_atm_eq[invVel3D[l]] - current_cell->f[invVel3D[l]];
//						}
//						else if (nb_cell->type == cellType::CT_OBSTACLE)
//						{
//							current_cell->ftemp[l] = current_cell->f[invVel3D[l]];
//						}
//					}
//
//					float3 normal = calculateNormal(i, j);
//					// Reconstruct atmospheric dfs for directions along the surface normal
//					for (int l = 0; l < 9; l++)
//					{
//						if (dot(normal, vel3Dv[invVel3D[l]]) > 0.f)
//							current_cell->ftemp[l] = f_atm_eq[l] + f_atm_eq[invVel3D[l]] - current_cell->f[invVel3D[l]];
//					}
//				}
//			}
//		}
//	}
//
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			for (int l = 0; l < 9; l++)
//				current_cell->f[l] = current_cell->ftemp[l];
//		}
//	}
//}
//
//void d3q19_lattice::collide()
//{
//	for (int i = 0; i < width; i++)
//	for (int j = 0; j < height; j++)
//	{
//		d3q19_cell* current_cell = getCellAt(i, j);
//
//		if (current_cell->type != cellType::CT_OBSTACLE && current_cell->type != cellType::CT_EMPTY)
//		{
//			current_cell->deriveQuantities(vMax);
//			current_cell->mass = current_cell->mass_temp;
//			
//			if (current_cell->type == cellType::CT_FLUID)
//				current_cell->mass = current_cell->rho;
//
//			float feq[9] = { 0 };
//			calculateEquilibrium(current_cell->velocity, current_cell->rho, feq);
//
//			int test = 0;
//			for (int l = 0; l < 9; l++)
//			{
//				current_cell->f[l] = (1.0f - w) * current_cell->f[l] + w * feq[l]
//					//	//To include gravity
//					+ weights3D[l] * current_cell->rho * dot(vel3Dv[l], float3{ latticeAcceleration, 0 });
//			}
//			
//			int test1 = 0;
//		}
//	}
//}
//
//void d3q19_lattice::updateCells()
//{
//	float k = 0.001f;
//
//#pragma region	// check whether interface cells emptied or filled
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			if (current_cell->type == cellType::CT_INTERFACE)
//			{
//				// Interfase cell filled
//				if (current_cell->mass > ((1.f + k) * current_cell->rho))
//					current_cell->type = cellType::CT_IF_TO_FLUID;
//				else if (current_cell->mass < ((0.f - k) * current_cell->rho))
//					current_cell->type = cellType::CT_IF_TO_EMPTY;
//			}
//		}
//	}
//#pragma endregion
//
//#pragma region	// set flags for filled interface cells (interface to fluid)
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			// Initialize values for empty cells
//			if (current_cell->type == cellType::CT_IF_TO_FLUID)
//			{
//				for (int l = 0; l < 9; l++)
//				{
//					d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//
//					if (nb_cell->type == cellType::CT_EMPTY)
//					{
//						nb_cell->type = cellType::CT_INTERFACE;
//						averageSurroundings(nb_cell, i - vel3Di[l].x, j - vel3Di[l].y);
//					}
//				}
//				
//				// prevent neighboring cells from becoming empty
//				for (int l = 0; l < 9; l++)
//				{
//					d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//
//					if (nb_cell->type == cellType::CT_IF_TO_EMPTY)
//						nb_cell->type = cellType::CT_INTERFACE;
//				}
//			}
//		}
//	}
//#pragma endregion
//
//#pragma region	// set flags for emptied interface cells (interface to empty)
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			// convert neighboring fluid cells to interface cells
//			if (current_cell->type == cellType::CT_IF_TO_EMPTY)
//			{
//				for (int l = 0; l < 9; l++)
//				{
//					d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//
//					if (nb_cell->type == cellType::CT_FLUID)
//						nb_cell->type = cellType::CT_INTERFACE;
//				}
//			}
//		}
//	}
//#pragma endregion
//
//#pragma region // distribute excess mass
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//			float3 normal = calculateNormal(i, j);
//			float excess_mass = 0.f;
//
//			if (current_cell->type == cellType::CT_IF_TO_FLUID)
//			{
//				excess_mass = current_cell->mass - current_cell->rho;
//				current_cell->mass = current_cell->rho;
//			}
//			else if (current_cell->type == cellType::CT_IF_TO_EMPTY)
//			{
//				excess_mass = current_cell->mass;
//				normal = float3_ScalarMultiply(-1.f, normal);
//				current_cell->mass = 0.f;
//			}
//			else
//			{
//
//				continue;
//			}
//			
//
//			float eta[9] = { 0.f };
//			float eta_total = 0.f;
//			unsigned int isIF[9] = { 0 };
//			unsigned int numIF = 0;	// number of interface cell neighbors
//
//			for (int l = 0; l < 9; l++)
//			{
//				// neighbor cell in the direction of velocity vector
//				d3q19_cell* nb_cell = getCellAt_Mod(i + vel3Di[l].x, j + vel3Di[l].y);
//				if (nb_cell->type == cellType::CT_INTERFACE)
//				{
//					eta[l] = dot(vel3Di[l], normal);
//					if (eta[l] < 0) eta[l] = 0;
//
//					eta_total += eta[l];
//					isIF[l] = 1;
//					numIF++;
//				}
//			}
//
//			// store excess mass to be distributed in ftemp
//			if (eta_total > 0)
//			{
//				float eta_fraction = 1 / eta_total;
//				for (int l = 0; l < 9; l++)
//					current_cell->mex[l] = excess_mass * eta[l] * eta_fraction;
//			}
//			//else if (numIF > 0)
//			//{
//			//	float excess_mass_uniform = excess_mass / numIF;
//			//	for (int l = 0; l < 9; l++)
//			//		current_cell->ftemp[l] = isIF[l] ? excess_mass_uniform : 0.f;
//			//}
//		}
//	}
//#pragma endregion
//
//#pragma region // collect distributed mass and finalize cell flags
//	for (int i = 0; i < width - 1; i++)
//	{
//		for (int j = 0; j < height - 1; j++)
//		{
//			d3q19_cell* current_cell = getCellAt(i, j);
//
//			if (current_cell->type == cellType::CT_INTERFACE)
//			{
//				for (int l = 0; l < 9; l++)
//				{
//					d3q19_cell* nb_cell = getCellAt_Mod(i - vel3Di[l].x, j - vel3Di[l].y);
//					nb_cell->mass += current_cell->mex[l];
//					current_cell->mex[l] = 0.f;
//				}
//			}
//			else if (current_cell->type == cellType::CT_IF_TO_FLUID)
//				current_cell->type = cellType::CT_FLUID;
//			else if (current_cell->type == cellType::CT_IF_TO_EMPTY)
//				current_cell->type = cellType::CT_EMPTY;
//		}
//	}
//#pragma endregion
//}
//
//void d3q19_lattice::averageSurroundings(d3q19_cell* cell_, int x_, int y_)
//{
//	float counter = 0;
//	cell_->mass = cell_->mass_temp = 0.f;
//	cell_->rho = 0.f;
//	cell_->velocity = float3{ 0.f, 0.f };
//
//	for (int l = 0; l < 9; l++)
//	{
//		d3q19_cell* nb_cell = getCellAt_Mod(x_ - vel3Di[l].x, y_ - vel3Di[l].y);
//
//		if (nb_cell->type == cellType::CT_FLUID || nb_cell->type == cellType::CT_INTERFACE)
//		{
//			cell_->rho += nb_cell->rho;
//			cell_->velocity += nb_cell->velocity;
//			counter += 1.f;
//		}
//	}
//	if (counter > 0)
//	{
//		cell_->rho /= counter;
//		cell_->velocity /= counter;
//	}
//
//	calculateEquilibrium(cell_->velocity, cell_->rho, cell_->f);
//}
//
//void d3q19_lattice::calculateEquilibrium(float3 velocity_, float rho_, float feq_[9])
//{
//	float eiU = 0;	// Dot product between speed direction and velocity
//	float eiUsq = 0; // Dot product squared
//	float uSq = dot(velocity_, velocity_);	//Velocity squared
//
//	for (int i = 0; i<stride; i++)
//	{
//		eiU = dot(vel3Dv[i], velocity_);
//		eiUsq = eiU * eiU;
//
//		//feq_[i] = weights3D[i] * (rho_ + (3.0f * eiU) - (1.5f * uSq) + (4.5f * eiUsq));
//		feq_[i] = weights3D[i] *  rho_ * (1.0f  + (3.0f * eiU) - (1.5f * uSq) + (4.5f * eiUsq));
//	}
//}
//
//void d3q19_lattice::initCells(int **typeArray_, float initRho_, float3 initVelocity_)
//{
//	float feq[9] = { 0 };
//	d3q19_cell *current_cell;
//	calculateEquilibrium(initVelocity_, initRho_, feq);
//
//	for (int i = 0; i < width; i++)
//	for (int j = 0; j < height; j++)
//	{
//		current_cell = getCellAt(i, j);
//		current_cell->type = typeArray_[i][j];
//		if (current_cell->type == cellType::CT_FLUID || current_cell->type == cellType::CT_INTERFACE)
//		{
//			for (int l = 0; l < 9; l++)
//				current_cell->f[l] = current_cell->ftemp[l] = feq[l];
//			/*memcpy(&(current_cell->f), feq, sizeof(float)* 9);*/
//
//			current_cell->deriveQuantities(vMax);
//
//			if (current_cell->type == cellType::CT_FLUID)
//				current_cell->mass = current_cell->mass_temp = current_cell->rho;
//
//			// (arbitrarily) assign half-filled mass
//			if (current_cell->type == cellType::CT_INTERFACE)
//				current_cell->mass = current_cell->mass_temp = 0.5f * current_cell->rho;
//		}
//	}
//}
//
//float3 d3q19_lattice::calculateNormal(int x_, int y_)
//{
//	float3 normal;
//	d3q19_cell* cell_1 = getCellAt(x_ - 1, y_);
//	d3q19_cell* cell_2 = getCellAt(x_ + 1, y_);
//	d3q19_cell* cell_3 = getCellAt(x_, y_ - 1);
//	d3q19_cell* cell_4 = getCellAt(x_, y_ + 1);
//
//	normal.x = 0.5f * (getCellAt_Mod(x_ - 1, y_)->calculateEpsilon() - getCellAt_Mod(x_ + 1, y_)->calculateEpsilon());
//	normal.y = 0.5f * (getCellAt_Mod(x_, y_ - 1)->calculateEpsilon() - getCellAt_Mod(x_, y_ + 1)->calculateEpsilon());
//	return normal;
//}
//
//// Basic 2d_LBM
////#include "Lattice_2D.h"
////#include <memory>
////
////d3q19_cell::d3q19_cell() : mass(0.0f), type(cellType::CT_EMPTY), rho(0.0f), velocity(float3{ 0.0f, 0.0f })
////{
////	//f = new float[9]();
////	//ftemp = new float[9]();
////	for (int i = 0; i < 9; i++)
////		f[i] = ftemp[i] = 0;
////}
////
////d3q19_cell::d3q19_cell(float *f_, float mass_ = 0.0f, float rho_ = 0.0f,
////	int type_ = cellType::CT_EMPTY, float3 velocity_ = float3{ 0.0f, 0.0f })
////{
////	for (int i = 0; i < 9; i++)
////		f[i] = ftemp[i] = f_[i];
////	//f = new float[9]();
////	//ftemp = new float[9]();
////	//memcpy(&f, f_, sizeof(float)* 9);
////	type = type_;
////	rho = rho_;
////	mass = mass_;
////	velocity = velocity_;
////}
////
////void d3q19_cell::deriveQuantities(float vMax_)
////{
////	rho = 0;
////	velocity = float3{ 0.0f, 0.0f };
////
////	for (int i = 0; i < 9; i++)
////		rho += f[i];
////
////	if (rho > 0.0f)
////	{
////		for (int i = 0; i < 9; i++)
////			velocity += float3_ScalarMultiply(f[i], vel3Dv[i]);
////
////		velocity = float3_ScalarMultiply(1.0f / rho, velocity);
////	}
////
////	float norm = float3_Norm(velocity);
////	if (norm > vMax_)
////		velocity = float3_ScalarMultiply(vMax_ / norm, velocity);
////}
////
////d3q19_lattice::d3q19_lattice(int width_, int height_, float worldViscosity_, float cellsPerSide_, float domainSize_)
////{
////	width = width_;
////	height = height_;
////	stride = 9;
////
////	// The cells are initiated statically with the predefined size. This initialization step would be needed with dynamic pointer
////	//cells = new d3q19_cell*[width];
////	//for (int i = 0; i < width; i++)
////	//	cells[i] = new d3q19_cell[height];
////
////	// Values for stability and to integrate gravity
////	c = (float)(1.0 / sqrt(3.0));
////	domainSize = domainSize_;
////	cellsPerSide = cellsPerSide_;
////	cellSize = domainSize_ / cellsPerSide_;
////	gravity = 9.8f;
////	timeStep = (float)(sqrtf((0.005f * cellSize) / fabs(gravity)));
////	vMax = cellSize / timeStep;
////	viscosity = worldViscosity_ * timeStep / (cellSize * cellSize);
////	tau = 3.0f * viscosity + 0.5f;
////	w = 1.0f / tau;
////	latticeAcceleration = gravity * timeStep * timeStep / cellSize;
////}
////
////void d3q19_lattice::step()
////{
////	stream();
////	collide();
////}
////
////void d3q19_lattice::stream()
////{
////	for (int i = 1; i < width - 1; i++)
////	{
////		for (int j = 1; j < height - 1; j++)
////		{
////			d3q19_cell* current_cell = getCellAt(i, j);
////
////			if (current_cell->type != cellType::CT_OBSTACLE || current_cell->type != cellType::CT_EMPTY)
////			{
////				for (int l = 0; l < 9; l++)
////				{
////					// f'i(x, t+dt) = fi(x+e�, t)
////					d3q19_cell* nb_cell = getCellAt(i - vel3Di[l].x, j - vel3Di[l].y);
////
////					if (nb_cell->type == cellType::CT_FLUID || nb_cell->type == cellType::CT_INTERFACE)
////					{
////						current_cell->ftemp[l] = nb_cell->f[l];
////					}
////					else if (nb_cell->type == cellType::CT_OBSTACLE)
////					{
////						current_cell->ftemp[l] = current_cell->f[invVel3D[l]];
////					}
////				}
////			}
////		}
////	}
////}
////
////void d3q19_lattice::collide()
////{
////	float *feq = new float[9]();
////
////	for (int i = 0; i < width; i++)
////	for (int j = 0; j < height; j++)
////	{
////		d3q19_cell* current_cell = getCellAt(i, j);
////
////		if (current_cell->type != cellType::CT_OBSTACLE && current_cell->type != cellType::CT_EMPTY)
////		{
////			current_cell->deriveQuantities(vMax);
////			calculateEquilibrium(current_cell->velocity, current_cell->rho, feq);
////
////			for (int l = 0; l < 9; l++)
////			{
////				current_cell->f[l] = (1.0f - w) * current_cell->ftemp[l] + w * feq[l]
////					//	//To include gravity
////					+ weights3D[l] * current_cell->rho * dot(vel3Dv[l], float3{ 0, -latticeAcceleration });
////			}
////		}
////	}
////}
////
////void d3q19_lattice::calculateEquilibrium(float3 velocity_, float rho_, float *feq_)
////{
////	float eiU = 0;	// Dot product between speed direction and velocity
////	float eiUsq = 0; // Dot product squared
////	float uSq = dot(velocity_, velocity_);	//Velocity squared
////
////	for (int i = 0; i<stride; i++)
////	{
////		eiU = dot(vel3Dv[i], velocity_);
////		eiUsq = eiU * eiU;
////
////		//feq_[i] = weights3D[i] * (rho_ + (3.0f * eiU) - (1.5f * uSq) + (4.5f * eiUsq));
////		feq_[i] = weights3D[i] * rho_ * (1.0f + (3.0f * eiU) - (1.5f * uSq) + (4.5f * eiUsq));
////	}
////}
////
////void d3q19_lattice::initCells(int **typeArray_, float initRho_, float3 initVelocity_)
////{
////	float *feq = new float[9]();
////	d3q19_cell *current_cell;
////	calculateEquilibrium(initVelocity_, initRho_, feq);
////
////	for (int i = 0; i < width; i++)
////	for (int j = 0; j < height; j++)
////	{
////		current_cell = getCellAt(i, j);
////		current_cell->type = typeArray_[i][j];
////		if (current_cell->type == cellType::CT_FLUID || current_cell->type == cellType::CT_INTERFACE)
////		{
////			for (int l = 0; l < 9; l++)
////				current_cell->f[l] = current_cell->ftemp[l] = feq[l];
////			/*memcpy(&(current_cell->f), feq, sizeof(float)* 9);*/
////
////			if (current_cell->type == cellType::CT_FLUID)
////				current_cell->mass = current_cell->rho;
////			// (arbitrarily) assign half-filled mass
////			if (current_cell->type == cellType::CT_INTERFACE)
////				current_cell->mass = 0.5f * current_cell->rho;
////		}
////
////		current_cell->deriveQuantities(vMax);
////	}
////}