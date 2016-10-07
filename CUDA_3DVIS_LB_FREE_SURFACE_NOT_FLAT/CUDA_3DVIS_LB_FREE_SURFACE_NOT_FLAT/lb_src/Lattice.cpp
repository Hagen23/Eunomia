#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float worldViscosity, float mass, float cellsPerSide, float domainSize)
{
	_width = width; _height = height; _depth = depth; _stride = 19;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd3q19[_numberElements]();
	c = 1.0f / sqrtf(3.0);

	_cellsPerSide = cellsPerSide;
	_domainSize = domainSize;

	cellSize = _domainSize / _cellsPerSide;

	gravity = 9.8f;

	timeStep = (float)(sqrtf((0.005f * cellSize) / fabs(gravity)));

	_vMax = cellSize / timeStep;

	viscosity = worldViscosity * timeStep / (cellSize * cellSize);

	_tau = 3.0f * viscosity + 0.5f;

	_w = 1 / _tau;

	latticeAcceleration = gravity * timeStep * timeStep / cellSize;
}
	
latticed3q19::~latticed3q19()
{
	delete[] latticeElements;
}

void latticed3q19::step(void)
{
	setNeighborhoodFlags();
	stream();
	collide();
	cellFlagReinitialization();
	//applyBoundaryConditions();
}

void latticed3q19::setNeighborhoodFlags()
{
	int nb_x, nb_y, nb_z;
	int nb_cellIndex;

	for (int k = 0; k<_depth; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i < _width; i++)
	{
		int cellIndex = I3D(_width, _height, i, j, k);

		if (latticeElements[cellIndex].isSolid)
			continue;

		latticeElements[cellIndex].cellType |= (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);

		for (int l = 1; l < _stride; l++)
		{
			nb_x = (int)(i + speedDirection[l].x);
			nb_y = (int)(j + speedDirection[l].y);
			nb_z = (int)(k + speedDirection[l].z);

			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if ((latticeElements[nb_cellIndex].cellType & cell_types::fluid) == cell_types::fluid)
				latticeElements[cellIndex].cellType &= ~CT_NO_FLUID_NEIGH;

			if ((latticeElements[nb_cellIndex].cellType & cell_types::gas) == cell_types::gas)
				latticeElements[cellIndex].cellType &= ~CT_NO_EMPTY_NEIGH;

			if ((latticeElements[nb_cellIndex].cellType & cell_types::interphase) == cell_types::interphase)
				latticeElements[cellIndex].cellType &= ~CT_NO_IFACE_NEIGH;
		}

		//if (latticeElements[cellIndex].cellType & (cell_types::interphase) == cell_types::interphase)
		//if(latticeElements[cellIndex].cellType & (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH)
		//	== (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH))
		//	int test = 0;

		// both flags should not be set simultaneously
		if ((latticeElements[cellIndex].cellType & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			latticeElements[cellIndex].cellType &= ~CT_NO_FLUID_NEIGH;
	}
}

float latticed3q19::calculateMassExchange(latticeElementd3q19 currentCell, latticeElementd3q19 neighborCell, int l)
{
	// Table 4.1 in Nils Thuerey's PhD thesis

	if ((currentCell.cellType & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
	{
		if ((neighborCell.cellType & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
			return neighborCell.f[inverseSpeedDirectionIndex[l]] - currentCell.f[l];
		else
			// neighbor is standard cell or CT_NO_EMPTY_NEIGH
			return -1.0f *  currentCell.f[l];
	}
	else if ((currentCell.cellType & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
	{
		if ((neighborCell.cellType & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			return neighborCell.f[inverseSpeedDirectionIndex[l]] - currentCell.f[l];
		else
			// neighbor is standard cell or CT_NO_FLUID_NEIGH
			return neighborCell.f[inverseSpeedDirectionIndex[l]];
	}
	else
	{
		// current cell is standard cell
		if ((neighborCell.cellType & CT_NO_FLUID_NEIGH) == CT_NO_FLUID_NEIGH)
			return neighborCell.f[inverseSpeedDirectionIndex[l]];
		else if ((neighborCell.cellType & CT_NO_EMPTY_NEIGH) == CT_NO_EMPTY_NEIGH)
			return -1.0f *  currentCell.f[l];
		else
			// neighbor is standard cell
			return neighborCell.f[inverseSpeedDirectionIndex[l]] - currentCell.f[l];
	}
}

void latticed3q19::calculateAirEquilibriumFunction(vector3d velocity, float *feq)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = velocity.dotProduct(velocity);	//Velocity squared

	for (int i = 0; i<_stride; i++)
	{
		w = latticeWeights[i];
		eiU = speedDirection[i].dotProduct(velocity);
		eiUsq = eiU * eiU;

		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		feq[i] = w * 1.0f * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

vector3d latticed3q19::calculateNormal(int i, int j, int k)
{
	return vector3d
	{
	(latticeElements[(I3D(_width, _height, i - 1, j, k))].getEpsilon() - latticeElements[(I3D(_width, _height, i + 1, j, k))].getEpsilon()) * 0.5f,
	(latticeElements[(I3D(_width, _height, i , j - 1, k))].getEpsilon() - latticeElements[(I3D(_width, _height, i, j + 1, k))].getEpsilon()) * 0.5f,
	(latticeElements[(I3D(_width, _height, i , j, k - 1))].getEpsilon() - latticeElements[(I3D(_width, _height, i, j, k + 1))].getEpsilon()) * 0.5f
	};
}

void latticed3q19::averageSurroundings(int i, int j, int k)
{
	int nb_x, nb_y, nb_z, numberOfCells = 0;
	vector3d averageVelocity = { 0.0f, 0.0f, 0.0f };
	float averageRo = 0;
	latticeElementd3q19 *currentCell, *neighborCell;

	currentCell = &latticeElements[I3D(_width, _height, i, j, k)];

	for (int l = 1; l < _stride; l++)
	{
		nb_x = (int)(i + speedDirection[l].x);
		nb_y = (int)(j + speedDirection[l].y);
		nb_z = (int)(k + speedDirection[l].z);

		if (nb_x < _width && nb_y < _height && nb_z < _depth)
		{
			neighborCell = &latticeElements[I3D(_width, _height, nb_x, nb_y, nb_z)];

			if ((neighborCell->cellType & cell_types::fluid) == cell_types::fluid
				|| (neighborCell->cellType & cell_types::interphase) == cell_types::interphase)
			{
				averageVelocity += neighborCell->velocityVector;
				averageRo += neighborCell->ro;
				numberOfCells++;
			}
		}
	}
	if (numberOfCells > 0)
	{
		averageVelocity /= (float)numberOfCells;
		averageRo /= numberOfCells;
		currentCell->calculateEquilibriumFunction(averageVelocity, averageRo);
		for (int l = 0; l < _stride; l++)
			currentCell->f[l] = currentCell->feq[l];
	}
}

void latticed3q19::cellFlagReinitialization()
{
	int nb_x, nb_y, nb_z;
	float excessMass = 0.0f, deltaMass = 0.0f;
	float *eta, etaTotal, etaFracc;

	//Type adjustment
	for (vector3d filledCell : _filledCells)
	{
		latticeElementd3q19 *currentCell = &latticeElements[I3D(_width, _height, (int)filledCell.x, (int)filledCell.y, (int)filledCell.z)];

		for (int l = 1; l < _stride; l++)
		{
			nb_x = (int)(filledCell.x + speedDirection[l].x);
			nb_y = (int)(filledCell.y + speedDirection[l].y);
			nb_z = (int)(filledCell.z + speedDirection[l].z);

			latticeElementd3q19 *neighborCell = &latticeElements[I3D(_width, _height, nb_x, nb_y, nb_z)];

			// Preparing the nighborhood of filled cells. All neighboring empty cells are
			//converted to interface cells.
			if ((neighborCell->cellType & cell_types::gas) == cell_types::gas)
			{
				neighborCell->cellTypeTemp = cell_types::interphase;
				averageSurroundings(nb_x, nb_y, nb_z);
			}
			else if ((neighborCell->cellType & cell_types::interphase) == cell_types::interphase)
			{
				std::vector<vector3d>::iterator emptyIterator =
					std::find(_emptiedCells.begin(), _emptiedCells.end(), vector3d{ (float)nb_x, (float)nb_y, (float)nb_z });

				if (emptyIterator != _emptiedCells.end())
					_emptiedCells.erase(emptyIterator);
			}
		}
		currentCell->cellTypeTemp = cell_types::fluid;
	}

	for (vector3d emptiedCell : _emptiedCells)
	{
		latticeElementd3q19 *currentCell = &latticeElements[I3D(_width, _height, (int)emptiedCell.x, (int)emptiedCell.y, (int)emptiedCell.z)];

		for (int l = 1; l < _stride; l++)
		{
			nb_x = (int)(emptiedCell.x + speedDirection[l].x);
			nb_y = (int)(emptiedCell.y + speedDirection[l].y);
			nb_z = (int)(emptiedCell.z + speedDirection[l].z);

			latticeElementd3q19 *neighborCell = &latticeElements[I3D(_width, _height, nb_x, nb_y, nb_z)];

			if ((neighborCell->cellType & cell_types::fluid) == cell_types::fluid)
				neighborCell->cellTypeTemp = cell_types::interphase;
		}

		currentCell->cellTypeTemp = cell_types::gas;
	}

	// Excess mass distribution
	for (vector3d filledCell : _filledCells)
	{
		excessMass = 0.0f;
		eta = new float[_stride]();
		etaTotal = 0.0f;
		deltaMass = 0.0f;

		latticeElementd3q19 *currentCell = &latticeElements[I3D(_width, _height, (int)filledCell.x, (int)filledCell.y, (int)filledCell.z)];

		if (currentCell->cellMass > currentCell->ro)
		{
			excessMass = currentCell->cellMass - currentCell->ro;
			vector3d normal = calculateNormal((int)filledCell.x, (int)filledCell.y, (int)filledCell.z);

			for (int l = 0; l < _stride; l++)
			{
				eta[l] = dot(normal, speedDirection[l]);
				if (eta[l] <= 0) eta[l] = 0;

				etaTotal += eta[l];
			}

			etaFracc = 1 / etaTotal;

			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(filledCell.x + speedDirection[l].x);
				nb_y = (int)(filledCell.y + speedDirection[l].y);
				nb_z = (int)(filledCell.z + speedDirection[l].z);

				latticeElementd3q19 *neighborCell = &latticeElements[I3D(_width, _height, nb_x, nb_y, nb_z)];

				if ((neighborCell->cellTypeTemp & cell_types::interphase) == cell_types::interphase)
					neighborCell->cellMassTemp += excessMass * eta[l] * etaFracc;
			}
		}
	}

	for (vector3d emptiedCell : _emptiedCells)
	{
		excessMass = 0.0f;
		eta = new float[_stride]();
		etaTotal = 0;

		latticeElementd3q19 *currentCell = &latticeElements[I3D(_width, _height, (int)emptiedCell.x, (int)emptiedCell.y, (int)emptiedCell.z)];

		if (currentCell->cellMass < 0.0f)
		{
			excessMass = currentCell->cellMass;
			vector3d normal = calculateNormal((int)emptiedCell.x, (int)emptiedCell.y, (int)emptiedCell.z);
			normal *= -1.0f;

			for (int l = 0; l < _stride; l++)
			{
				eta[l] = dot(normal, speedDirection[l]);
				if (eta[l] >= 0) eta[l] = 0;

				etaTotal += eta[l];
			}

			etaFracc = 1 / etaTotal;

			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(emptiedCell.x + speedDirection[l].x);
				nb_y = (int)(emptiedCell.y + speedDirection[l].y);
				nb_z = (int)(emptiedCell.z + speedDirection[l].z);

				latticeElementd3q19 *neighborCell = &latticeElements[I3D(_width, _height, nb_x, nb_y, nb_z)];

				if ((neighborCell->cellTypeTemp & cell_types::interphase) == cell_types::interphase)
					neighborCell->cellMassTemp += excessMass * eta[l] * etaFracc;
			}
		}
	}

	for (int i = 0; i < _numberElements; i++)
	{
		//for (int j = 0; j < 19; j++)
		//	latticeElements[i].f[j] = latticeElements[i].ftemp[j];
		
		latticeElements[i].cellType = latticeElements[i].cellTypeTemp;
		latticeElements[i].cellMass = latticeElements[i].cellMassTemp;

		latticeElements[i].cellType &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
		latticeElements[i].cellTypeTemp &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);

		if ((latticeElements[i].cellType & cell_types::fluid) == cell_types::fluid)
			latticeElements[i].cellMass = latticeElements[i].ro;

		if ((latticeElements[i].cellType & cell_types::gas) == cell_types::gas)
			latticeElements[i].cellMass = 0;

		latticeElements[i].cellMassTemp = 0;
		latticeElements[i].getEpsilon();
	}

	_filledCells.clear();
	_emptiedCells.clear();
}

void latticed3q19::stream()
{
	int cellIndex, neighborIndex, newI, newJ, newK;
	float *deltaMass = new float[_stride](), totalDelta = 0.0f;

	for(int k =0; k<_depth; k++)
	for(int j = 0; j < _height; j++)
	for(int i = 0; i<_width; i++)
	{
		cellIndex = I3D(_width, _height, i, j, k);

		memset(deltaMass, 0, sizeof(float)* 19);

		latticeElementd3q19 *currentCell = &latticeElements[cellIndex];

		if (!currentCell->isSolid)
		{
			if ((currentCell->cellType & cell_types::fluid) == cell_types::fluid)
			{
				for (int l = 1; l < 19; l++)
				{
					newI = (int)(i + speedDirection[l].x);
					newJ = (int)(j + speedDirection[l].y);
					newK = (int)(k + speedDirection[l].z);

					neighborIndex = I3D(_width, _height, newI, newJ, newK);

					if ((latticeElements[neighborIndex].cellType & cell_types::fluid) == cell_types::fluid 
						|| (latticeElements[neighborIndex].cellType & cell_types::interphase) == cell_types::interphase)
					currentCell->ftemp[l] = latticeElements[neighborIndex].f[l];
				}
			}
			if ((currentCell->cellType & cell_types::interphase) == cell_types::interphase)
			{
				float currentEpsilon = currentCell->getEpsilon();
				float *feqAir = new float[_stride]();
				//deltaMass = new float[_stride]();
				//memset(deltaMass, 0, sizeof(float)* 19);
				calculateAirEquilibriumFunction(currentCell->velocityVector, feqAir);
				vector3d normal = calculateNormal(i, j, k);

				for (int l = 1; l < 19; l++)
				{
					if (deltaMass != deltaMass)
						int test = 0;

					newI = (int)(i + speedDirection[l].x);
					newJ = (int)(j + speedDirection[l].y);
					newK = (int)(k + speedDirection[l].z);

					neighborIndex = I3D(_width, _height, newI, newJ, newK);

					latticeElementd3q19 neighborCell = latticeElements[neighborIndex];

					if ((neighborCell.cellType & cell_types::fluid) == cell_types::fluid)
					{
						deltaMass[l] = neighborCell.f[inverseSpeedDirectionIndex[l]] - currentCell->f[l];
						//deltaMass += neighborCell.f[inverseSpeedDirectionIndex[l]] - currentCell->f[l];
						currentCell->ftemp[l] = neighborCell.f[l];
					}
					else if ((neighborCell.cellType & cell_types::interphase) == cell_types::interphase)
					{
						float neighborEpsilon = neighborCell.getEpsilon();
						float massExchange = calculateMassExchange(*currentCell, neighborCell, l);
						if (massExchange > 0)
							int test = 0;
						deltaMass[l] = massExchange * (currentEpsilon + neighborEpsilon) * 0.5f;
						//deltaMass += calculateMassExchange(*currentCell, neighborCell, l) * (currentEpsilon + neighborEpsilon) * 0.5f;
						currentCell->ftemp[l] = neighborCell.f[l];
					}
					else if ((neighborCell.cellType & cell_types::gas) == cell_types::gas)
					{
						currentCell->ftemp[l] =
							feqAir[l] + feqAir[inverseSpeedDirectionIndex[l]] - currentCell->f[inverseSpeedDirectionIndex[l]];
					}

					//if (dot(normal, speedDirection[l]) < 0)
					if (dot(normal, inverseSpeedDirection[l]) > 0)
						currentCell->ftemp[l] = 
							feqAir[l] + feqAir[inverseSpeedDirectionIndex[l]] - currentCell->f[inverseSpeedDirectionIndex[l]];
				}

				if (deltaMass != deltaMass)
					int test = 0;

				for (int l = 1; l < 19; l++)
					totalDelta += deltaMass[l];

				currentCell->cellMassTemp = currentCell->cellMass + totalDelta;
			}
		}
	}

	for (int i = 0; i < _numberElements; i++)
	{
		for (int j = 0; j < 19; j++)
			latticeElements[i].f[j] = latticeElements[i].ftemp[j];

		//if (latticeElements[i].cellType & cell_types::interphase)
		//{
		//	latticeElements[i].cellMass = latticeElements[i].cellMassTemp;
		//	latticeElements[i].cellMassTemp = 0;
		//}
	}
	
}

void latticed3q19::setFilledOrEmptied(int i, int j, int k)
{
	int i0 = I3D(_width, _height, i, j, k);
	if (!latticeElements[i0].isSolid)
	{
		if ((latticeElements[i0].cellType & cell_types::interphase) == cell_types::interphase)
		{
			latticeElementd3q19 *currentCell = &latticeElements[i0];

			if (
				(currentCell->cellMassTemp > ((1.0f + FILL_OFFSET) * currentCell->ro)) // Eq. (4.7)
				|| // Remove artifacts
				((currentCell->cellMassTemp > ((1.0f - LONELY_THRESH)* currentCell->ro)) 
				&& ((currentCell->cellType & cell_types::CT_NO_FLUID_NEIGH) == cell_types::CT_NO_FLUID_NEIGH))
				)
			{
				// interface to fluid cell
				currentCell->cellTypeTemp = cell_types::CT_IF_TO_FLUID;
				_filledCells.push_back(vector3d{ (float)i, (float)j, (float)k });
			}
			else if (
				(currentCell->cellMassTemp < -0.001f * currentCell->ro) // Eq. (4.7)
				||
				((currentCell->cellMassTemp < 0.1f*currentCell->ro) 
				&& ((currentCell->cellType & CT_NO_FLUID_NEIGH) == cell_types::CT_NO_FLUID_NEIGH))
				)	// isolated interface cell: only empty or obstacle neighbors
			{
				// interface to empty cell
				currentCell->cellTypeTemp = cell_types::CT_IF_TO_EMPTY;
				_emptiedCells.push_back(vector3d{ (float)i, (float)j, (float)k });
			}
		}
	}
}

void latticed3q19::collide(void)
{
	for (int k = 0; k<_depth; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i<_width; i++)
	{
		int cellIndex = I3D(_width, _height, i, j, k);
		latticeElementd3q19 *currentCell = &latticeElements[cellIndex];

		if ((currentCell->cellType & cell_types::gas) == cell_types::gas)
			continue;

		if (!currentCell->isSolid)
		{
			if ((currentCell->cellType & (cell_types::fluid)) == (cell_types::fluid) ||
				(currentCell->cellType & (cell_types::interphase)) == (cell_types::interphase))
			{
				currentCell->calculateQuantities();
				currentCell->calculateEquilibriumFunction(currentCell->velocityVector, currentCell->ro);
				setFilledOrEmptied(i, j, k);

				for (int l = 1; l < 19; l++)
					currentCell->f[l] = (1.0f - _w) * currentCell->f[l] + _w * currentCell->feq[l]
					// For gravity
					+ latticeWeights[l] * currentCell->ro * dot(speedDirection[l], vector3d{ 0.0f, -latticeAcceleration, 0.0f });
				//latticeElements[i0].f[l] = latticeElements[i0].f[l] - (latticeElements[i0].f[l] - latticeElements[i0].feq[l]) / _tau;
				//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
			}
		}		
		else
			solid_BC(cellIndex);
	}
}

void latticed3q19::in_BC(vector3d inVector)
{
	for (int j = 0; j < _height; j++)
	{
		for (int i = 0; i < _width; i++)
		{
			int i0 = I3D(_width, _height, i, j, 24);
			latticeElements[i0].calculateInEquilibriumFunction(inVector, latticeElements[i0].getRo());
		}
	}
}

void latticed3q19::applyBoundaryConditions()
{
	//in_BC(vector3d(0.0,0.0, -0.6));
}

void latticed3q19::solid_BC(int i0)
{
	float temp;

	temp = latticeElements[i0].f[1]; 	latticeElements[i0].f[1] = latticeElements[i0].f[2];	latticeElements[i0].f[2] = temp;		// f1	<-> f2
	temp = latticeElements[i0].f[3];	latticeElements[i0].f[3] = latticeElements[i0].f[4];	latticeElements[i0].f[4] = temp;		// f3	<-> f4
	temp = latticeElements[i0].f[5];	latticeElements[i0].f[5] = latticeElements[i0].f[6];	latticeElements[i0].f[6] = temp;		// f5	<-> f6
	temp = latticeElements[i0].f[7];	latticeElements[i0].f[7] = latticeElements[i0].f[12];	latticeElements[i0].f[12] = temp;		// f7	<-> f12
	temp = latticeElements[i0].f[8];	latticeElements[i0].f[8] = latticeElements[i0].f[11];	latticeElements[i0].f[11] = temp;		// f8	<-> f11
	temp = latticeElements[i0].f[9];	latticeElements[i0].f[9] = latticeElements[i0].f[14];	latticeElements[i0].f[14] = temp;		// f9	<-> f14
	temp = latticeElements[i0].f[10];	latticeElements[i0].f[10] = latticeElements[i0].f[13];	latticeElements[i0].f[13] = temp;		// f10	<-> f13
	temp = latticeElements[i0].f[15];	latticeElements[i0].f[15] = latticeElements[i0].f[18];	latticeElements[i0].f[18] = temp;		// f15	<-> f18
	temp = latticeElements[i0].f[16];	latticeElements[i0].f[16] = latticeElements[i0].f[17];	latticeElements[i0].f[17] = temp;		// f16	<-> f17
}

//void latticed3q19::solid_BC(int i0)
//{
//	double temp;
//
//	temp = latticeElements[i0].f[1]; 	latticeElements[i0].f[1] = latticeElements[i0].f[3];	latticeElements[i0].f[3] = temp;		// f1	<-> f3
//	temp = latticeElements[i0].f[2];	latticeElements[i0].f[2] = latticeElements[i0].f[4];	latticeElements[i0].f[4] = temp;		// f2	<-> f4
//	temp = latticeElements[i0].f[5];	latticeElements[i0].f[5] = latticeElements[i0].f[6];	latticeElements[i0].f[6] = temp;		// f5	<-> f6
//	temp = latticeElements[i0].f[7];	latticeElements[i0].f[7] = latticeElements[i0].f[9];	latticeElements[i0].f[9] = temp;		// f7	<-> f9
//	temp = latticeElements[i0].f[8];	latticeElements[i0].f[8] = latticeElements[i0].f[10];	latticeElements[i0].f[10] = temp;		// f8	<-> f10
//	temp = latticeElements[i0].f[11];	latticeElements[i0].f[11] = latticeElements[i0].f[13];	latticeElements[i0].f[13] = temp;		// f11	<-> f13
//	temp = latticeElements[i0].f[12];	latticeElements[i0].f[12] = latticeElements[i0].f[14];	latticeElements[i0].f[14] = temp;		// f12	<-> f14
//	temp = latticeElements[i0].f[15];	latticeElements[i0].f[15] = latticeElements[i0].f[18];	latticeElements[i0].f[18] = temp;		// f15	<-> f18
//	temp = latticeElements[i0].f[16];	latticeElements[i0].f[16] = latticeElements[i0].f[17];	latticeElements[i0].f[17] = temp;		// f16	<-> f17
//}

void latticed3q19::printLattice(void)
{
	for(int i=0; i<_numberElements; i++)
	{
			latticeElements[i].printElement();
			std::cout << std::endl;
	}
}

void latticed3q19::printLatticeElement(int i, int j, int k)
{
	latticeElements[I3D(_width,_height, i,j,k)].printElement();
	std::cout << std::endl;
}

int latticed3q19::getNumElements(void)
{
	return _numberElements;
}

void latticed3q19::logInterfaseValues(std::ofstream &file, std::string message)
{
	if (file.is_open())
	{
		for (int k = 0; k < _depth; k++)
		for (int j = 0; j < _height; j++)
		for (int i = 0; i < _width; i++)
		{
			int cellIndex = I3D(_width, _height, i, j, k);
			latticeElementd3q19 *currentCell = &latticeElements[cellIndex];

			if (currentCell->cellType & cell_types::interphase)
			{
				std::string nb_status = "";
				if (currentCell->cellType & cell_types::CT_NO_EMPTY_NEIGH) nb_status += "CT_NO_EMPTY_NEIGH-";
				if (currentCell->cellType & cell_types::CT_NO_FLUID_NEIGH) nb_status += "CT_NO_FLUID_NEIGH-";
				if (currentCell->cellType & cell_types::CT_NO_IFACE_NEIGH) nb_status += "CT_NO_IFACE_NEIGH-";
				if (currentCell->cellType & cell_types::CT_IF_TO_EMPTY) nb_status += "CT_IF_TO_EMPTY-";
				if (currentCell->cellType & cell_types::CT_IF_TO_FLUID) nb_status += "CT_IF_TO_FLUID-";
				nb_status += ",";
				
				file << message + ",";
				file << std::to_string(cellIndex) + "," + nb_status
					+ std::to_string(currentCell->ro) + "," + std::to_string(currentCell->cellMass) + ","
					+ std::to_string(currentCell->cellMassTemp) + "," + std::to_string(currentCell->epsilon) + ",";

				for (int l = 0; l < _stride; l++)
					file << std::to_string(currentCell->f[l]) + ",";
				file << "\n";
			}
		}
	}
}