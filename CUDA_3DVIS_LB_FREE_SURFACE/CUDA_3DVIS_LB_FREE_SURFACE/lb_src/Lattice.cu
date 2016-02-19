#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float worldViscosity, float mass, float cellsPerSide, float domainSize)
{
	_width = width; _height = height; _depth = depth;
	_stride = 19;
	_numberLatticeElements = _width * _height * _depth;
	_numberAllElements = _stride * _numberLatticeElements;
	_cellsPerSide = cellsPerSide;
	_domainSize = domainSize;
	_mass = mass;

	f = new float[_numberAllElements]();
	ftemp = new float[_numberAllElements]();
	feq = new float[_stride]();
	solid = new unsigned int[_numberLatticeElements]();
	velocityVector = new float3[_numberLatticeElements]();
	ro = new float[_numberLatticeElements]();

	epsilon = new float[_numberLatticeElements]();
	cellType = new int[_numberLatticeElements];
	cellTypeTemp = new int[_numberLatticeElements];
	cellMass = new float[_numberLatticeElements]();
	cellMassTemp = new float[_numberLatticeElements]();

	for (int i = 0; i < _numberLatticeElements; i++)
		cellType[i] = cellTypeTemp[i] = cell_types::gas;

	c = (float)(1.0 / sqrt(3.0));
	
	// These values are needed to maintain the fluid stability. Part 3.3 of the reference Thesis
	cellSize = _domainSize / _cellsPerSide;

	gravity = 9.8f;

	timeStep = (float)(sqrtf( (0.005f * cellSize) / fabs(gravity)));

	_vMax = cellSize / timeStep;

	viscosity = worldViscosity * timeStep / (cellSize * cellSize);

	_tau = 3.0f * viscosity + 0.5f;

	_w = 1 / _tau;

	latticeAcceleration = gravity * timeStep * timeStep / cellSize;
}
	
latticed3q19::~latticed3q19()
{
	delete[] f;
	delete[] ftemp;
	delete[] feq;
	delete[] solid;
	delete[] ro;
	delete[] velocityVector;
	delete[] epsilon;
	delete[] cellType;
	delete[] cellMass;
	delete[] cellMassTemp;
}

void latticed3q19::step(void)
{
	// I have disabled the artifact removal sections until the free surface changes work.
	setNeighborhoodFlags();

	stream();

	collide();

	cellTypeAdjustment();

	//applyBoundaryConditions();
}

void latticed3q19::stream()
{
	int cellIndex, cellDf, advectedDf, advectedCell, inverseAdvectedDf;
	int newI, newJ, newK;
	float3 normal;
	float deltaMass = 0;

	for (int k = 0; k<_depth; k++)
	{
		for (int j = 0; j < _height; j++)
		{
			for (int i = 0; i<_width; i++)
			{
				cellIndex = I3D(_width, _height, i, j, k);

				if (solid[cellIndex] == 0)
				{
					if (cellType[cellIndex] & cell_types::fluid)
					{
						//cellTypeTemp[cellIndex] = cellType[cellIndex];

						try
						{
							if (cellMass[cellIndex] == ro[cellIndex])
							{
								for (int l = 0; l < 19; l++)
								{
									newI = (int)(i + speedDirection[l].x);
									newJ = (int)(j + speedDirection[l].y);
									newK = (int)(k + speedDirection[l].z);

									//newI = (int)(i + speedDirection[inverseSpeedDirectionIndex[l]].x);
									//newJ = (int)(j + speedDirection[inverseSpeedDirectionIndex[l]].y);
									//newK = (int)(k + speedDirection[inverseSpeedDirectionIndex[l]].z);

									cellDf = cellIndex * _stride + l;
									advectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, l);
									advectedCell = I3D(_width, _height, newI, newJ, newK);

									// Added step to calculate the mass exchange. Part 4.1 of the reference Thesis
									if (cellType[advectedCell] & (cell_types::fluid | cell_types::interphase)) //cell_types::fluid |
									{
										//inverseAdvectedDf = I3D_S(_width, _height, _stride,
										//	(int)(i + speedDirection[l].x), (int)(j + speedDirection[l].y), (int)(k + speedDirection[l].z),
										//	inverseSpeedDirectionIndex[l]);

										//// mass exchange between fluid and interface cell, Eq. (4.2)
										deltaMass += f[inverseAdvectedDf] - f[cellDf];
										//cellMassTemp[cellIndex] += f[inverseAdvectedDf] - f[cellDf];
										////cellMassTemp[cellIndex] += f[advectedDf] - f[cellIndex * _stride + inverseSpeedDirectionIndex[l]];

										ftemp[cellDf] = f[advectedDf];
									}								

								}
							}
							else
								throw std::string("Not the same mass (" +std::to_string(cellMass[cellIndex]) 
								+ ") and density (" + std::to_string(ro[cellIndex]) + ") at ") + std::to_string(cellIndex);
						}
						catch (std::string message) { cout << "Exception: " << message.c_str() << endl;		}
						catch (...){ cout << "Default exception." << endl; }
					}
					// Additional step to consider the interfase cells streaming and mass exchange. 
					else if (cellType[cellIndex] & cell_types::interphase)
					{
						float currentEpsilon = calculateEpsilon(cellIndex);
						
						deltaMass = 0;
						// Calculate air equilibrium function to reconstruct missing dfs
						calculateEquilibriumFunction(velocityVector[cellIndex], 1.0f);
						normal = calculateNormal(i, j, k);

						for (int l = 0; l < _stride; l++)
						{
							newI = (int)(i + speedDirection[l].x);
							newJ = (int)(j + speedDirection[l].y);
							newK = (int)(k + speedDirection[l].z);

							//newI = (int)(i + speedDirection[inverseSpeedDirectionIndex[l]].x);
							//newJ = (int)(j + speedDirection[inverseSpeedDirectionIndex[l]].y);
							//newK = (int)(k + speedDirection[inverseSpeedDirectionIndex[l]].z);
							
							cellDf = cellIndex*_stride + l;
							advectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, l);
							advectedCell = I3D(_width, _height, newI, newJ, newK);

							inverseAdvectedDf = I3D_S(_width, _height, _stride, 
								(int)(i + speedDirection[l].x), (int)(j + speedDirection[l].y), (int)(k + speedDirection[l].z),
								inverseSpeedDirectionIndex[l]);

							//For interface cells with neighboring fluid cells, the mass change has to conform to the
							//DFs exchanged during streaming, as fluid cells don’t require additional computations
							if (cellType[advectedCell] & cell_types::fluid)
							{
								// mass exchange between fluid and interface cell, Eq. (4.2)
								deltaMass += f[inverseAdvectedDf] - f[cellDf];
								//cellMassTemp[cellIndex] += f[inverseAdvectedDf] - f[cellDf];
								ftemp[cellDf] = f[advectedDf];
							}
							else if (cellType[advectedCell] & cell_types::interphase)
							{
								// mass exchange between two interface cells, Eq. (4.3)
								float neighborEpsilon = calculateEpsilon(advectedCell);
								float massExchange = calculateMassExchange(cellIndex, advectedCell, f[cellDf], f[inverseAdvectedDf]);
								
								//deltaMass += (f[inverseAdvectedDf] - f[cellDf]) * 0.5f * (currentEpsilon + neighborEpsilon);
								deltaMass += (massExchange)*0.5f * (currentEpsilon + neighborEpsilon);
								//cellMassTemp[cellIndex] += (f[inverseAdvectedDf] - f[cellDf]) * 0.5f * (currentEpsilon + neighborEpsilon);
								//cellMassTemp[cellIndex] += (f[advectedDf] - f[cellIndex * _stride + inverseSpeedDirectionIndex[l]]) 
								//	* 0.5f * (currentEpsilon + neighborEpsilon);
								
								// This line handles the cell artifacts, commented until the free surface changes work
								//cellMassTemp[cellIndex] += (massExchange)*0.5f * (currentEpsilon + neighborEpsilon);

								ftemp[cellDf] = f[advectedDf];
							}
							else if (cellType[advectedCell] & cell_types::gas) 	// Eq. (4.5)
							{
								ftemp[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
									feq[l] + feq[inverseSpeedDirectionIndex[l]]	- f[cellDf];
								//ftemp[cellDf] = feq[l] + feq[inverseSpeedDirectionIndex[l]] 
								//	- f[cellIndex * _stride + inverseSpeedDirectionIndex[l]];
							}
						}

						//// always use reconstructed atmospheric distribution function for directions along surface normal;
						//// separate loop to handle mass exchange correctly
						for (int l = 0; l < 19; l++)		
						{
							cellDf = cellIndex*_stride + l;

							if (dot(normal, speedDirection[l]) < 0)		// Eq. (4.6)
							//if (dot(normal, speedDirection[inverseSpeedDirectionIndex[l]]) > 0)		// Eq. (4.6)
							// reconstructed atmospheric distribution function, Eq. (4.5)
							//ftemp[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
							//feq[l] + feq[inverseSpeedDirectionIndex[l]] - f[cellDf];
							ftemp[cellDf] = feq[l] + feq[inverseSpeedDirectionIndex[l]]	- f[cellDf];
						}

						cellMassTemp[cellIndex] = cellMass[cellIndex] + deltaMass;
					}
				}
				else
				{
					// Cycle to bounce the dfs when an obstacle is encountered
					for (int l = 0; l < 19; l++)
					{
						cellDf = cellIndex*_stride + l;
						advectedDf = cellIndex*_stride + inverseSpeedDirectionIndex[l];

						ftemp[cellDf] = f[advectedDf];
					}
				}
			}
		}
	}

	// After the streaming steps, copy the dfs and the mass to the respective arrays
	for (int i0 = 0; i0 < _numberAllElements; i0++)
	{
		f[i0] = ftemp[i0];
		if (i0 % _stride == 0)
		{
			int index = i0 / _stride;

			if (cellType[index] & (cell_types::fluid | cell_types::interphase))
			{
				cellMass[index] = cellMassTemp[index];
				cellMassTemp[index] = 0;
				epsilon[index] = cellMass[index] / ro[index];
			}

			//if (cellType[index] & (cell_types::fluid))
			//{
			//	cellMass[index] = ro[index];
			//	epsilon[index] = 1.0f;
			//}
		}
	}
		
}

// Collision remains the same. Just added a step to determine if a given cell filled or emptied.
void latticed3q19::collide(void)
{
	int iBase = 0, numFluid = 0;
	float roAverage = 0.0f;
	//for (int i0 = 0; i0 < _numberLatticeElements; i0++)
	for (int k = 0; k<_depth; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i<_width; i++)
	{
		int i0 = I3D(_width, _height, i, j, k);
		if (solid[i0] == 0)
		{
			if (cellType[i0] & (cell_types::fluid | cell_types::interphase))
			{
				deriveQuantities(i0);

				setFilledOrEmpty(i, j, k);

				//velocityVector[i0].y += latticeAcceleration;

				calculateEquilibriumFunction(velocityVector[i0], ro[i0]);;

				for (int l = 0; l < 19; l++)
				{
					iBase = i0 * _stride + l;
					//f[iBase] = f[iBase] - (f[iBase] - feq[l]) / _tau;
					f[iBase] = (1.0f - _w) * f[iBase] + _w * feq[l]
						//To include gravity
						+ latticeWeights[l] * ro[i0] * dot(speedDirection[l], float3{ 0, -latticeAcceleration, 0 });
				}
				
				// Test that shows that the density of the fluid eventually becomes indeterminate; one of the errors I found
				roAverage += ro[i0];
				numFluid++;

				//if (cellMassTemp[i0] < 0.0f)
				//if (cellType[i0] & cell_types::fluid)
				//	cellMass[i0] = ro[i0];
				//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
			}
		}
		//else
		//	solid_BC(i0);
	}
	cout << "RO av: " << roAverage / numFluid << endl;
}

void latticed3q19::applyBoundaryConditions()
{
	//in_BC(vector3d(0.0,0.0, -0.6));
}

void latticed3q19::solid_BC(int i0)
{
	float temp;

	temp = f[i0*_stride + 1]; 	f[i0*_stride + 1] = f[i0*_stride + 2];		f[i0*_stride + 2] = temp;		// f1	<-> f2
	temp = f[i0*_stride + 3];	f[i0*_stride + 3] = f[i0*_stride + 4];		f[i0*_stride + 4] = temp;		// f3	<-> f4
	temp = f[i0*_stride + 5];	f[i0*_stride + 5] = f[i0*_stride + 6];		f[i0*_stride + 6] = temp;		// f5	<-> f6
	temp = f[i0*_stride + 7];	f[i0*_stride + 7] = f[i0*_stride + 12];		f[i0*_stride + 12] = temp;		// f7	<-> f12
	temp = f[i0*_stride + 8];	f[i0*_stride + 8] = f[i0*_stride + 11];		f[i0*_stride + 11] = temp;		// f8	<-> f11
	temp = f[i0*_stride + 9];	f[i0*_stride + 9] = f[i0*_stride + 14];		f[i0*_stride + 14] = temp;		// f9	<-> f14
	temp = f[i0*_stride + 10];	f[i0*_stride + 10] = f[i0*_stride + 13];	f[i0*_stride + 13] = temp;		// f10	<-> f13
	temp = f[i0*_stride + 15];	f[i0*_stride + 15] = f[i0*_stride + 18];	f[i0*_stride + 18] = temp;		// f15	<-> f18
	temp = f[i0*_stride + 16];	f[i0*_stride + 16] = f[i0*_stride + 17];	f[i0*_stride + 17] = temp;		// f16	<-> f17
}

//void latticed3q19::calculateSpeedVector(int index)
//{
//	//calculateRo();
//	//rovx = rovy = rovz = 0; 
//
//	ro[index] = rovx = rovy = rovz = 0;
//	int i0 = 0;
//	for (int i = 0; i<_stride; i++)
//	{
//		i0 = index * _stride + i;
//		ro[index] += f[i0];
//		rovx += f[i0] * speedDirection[i].x;
//		rovy += f[i0] * speedDirection[i].y;
//		rovz += f[i0] * speedDirection[i].z;
//	}
//
//	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
//	if (ro[index] == ro[index] && ro[index] != 0.0f)
//	{
//		velocityVector[index].x = rovx / ro[index];
//		velocityVector[index].y = rovy / ro[index];
//		velocityVector[index].z = rovz / ro[index];
//	}
//	else
//	{
//		velocityVector[index].x = 0;
//		velocityVector[index].y = 0;
//		velocityVector[index].z = 0;
//	}
//}

void latticed3q19::calculateEquilibriumFunction(float3 inVector, float inRo)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(inVector, inVector);	//Velocity squared

	for (int i = 0; i<_stride; i++)
	{
		w = latticeWeights[i];
		eiU = dot(speedDirection[i], inVector);
		eiUsq = eiU * eiU;
		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		feq[i] = w * inRo * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

// Have to find how to correctly initialize the interfase cells. 
void latticed3q19::calculateInEquilibriumFunction(int index, float3 inVector, float inRo)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = 0;	//Velocity squared
	
	uSq = dot(inVector, inVector);
	int iBase = 0;
	// For the interfase cells, an atmospheric pressure (1.0) is used as density
	ro[index] = cellType[index] & (cell_types::fluid) ? inRo :
		cellType[index] & (cell_types::interphase) ? inRo : 1.0f;

	if (cellType[index] & (cell_types::fluid | cell_types::interphase))
	{
		for (int i = 0; i<_stride; i++)
		{
			w = latticeWeights[i];
			eiU = dot(speedDirection[i], inVector);
			eiUsq = eiU * eiU;

			iBase = index*_stride + i;
			//ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
			ftemp[iBase] = f[iBase] = w; // *ro[index] * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
		}

		deriveQuantities(index);

		if (cellType[index] & cell_types::fluid)
		{
			cellMass[index] = cellMassTemp[index] = ro[index];
			epsilon[index] = cellMass[index] / ro[index];
		}
		else if (cellType[index] & cell_types::interphase)
		{
			// Arbitrarily assign very little mass to the interfase cells
			cellMass[index] = cellMassTemp[index] = 0.01f * ro[index];
			epsilon[index] = cellMass[index] / ro[index];
		}
	}
}

// Calculates the density and speed of a cell
void latticed3q19::deriveQuantities(int index)
{
	int dfIndex = 0;

	// Calculate average density
	ro[index] = 0;
	for (int l = 0; l < _stride; l++)
		ro[index] += f[index * _stride + l];

	velocityVector[index] = float3{ 0, 0, 0 };

	if (ro[index] >= 0)
	{
		for (int l = 0; l < _stride; l++)
		{
			dfIndex = index * _stride + l;
			velocityVector[index].x += f[dfIndex] * speedDirection[l].x;
			velocityVector[index].y += f[dfIndex] * speedDirection[l].y;
			velocityVector[index].z += f[dfIndex] * speedDirection[l].z;
		}
		velocityVector[index].x /= ro[index];
		velocityVector[index].y /= ro[index];
		velocityVector[index].z /= ro[index];
	}

	float n = float3_Norm(velocityVector[index]);
	if (n > _vMax)
		velocityVector[index] = float3_ScalarMultiply(_vMax / n, velocityVector[index]);
}

// Determines the initial mass of a cell. Still have to check this initialization...
void latticed3q19::calculateInitialMass()
{
	for (int i = 0; i < _numberLatticeElements; i++)
	{
		deriveQuantities(i);

		if (cellType[i] & cell_types::fluid)
		{
			cellMass[i] = cellMassTemp[i] = ro[i];
			epsilon[i] = cellMass[i] / ro[i];
		}
		else if(cellType[i] & cell_types::interphase)
		{
			cellMass[i] = cellMassTemp[i] = 0.01f * ro[i];
			epsilon[i] = cellMass[i] / ro[i];
		}
	}
}

float latticed3q19::calculateEpsilon(int index)
{
	if ((cellType[index] & cell_types::fluid) || (solid[index] == 1))
	{
		epsilon[index] = 1;
		return 1;
	}
	else if (cellType[index] & cell_types::interphase)
	{
		if (ro[index] > 0)
		{
			float epsilon_temp = cellMass[index] / ro[index];

			// df->mass can even be < 0 or > df->rho for interface cells to be converted to fluid or empty cells in the next step;
			// clamp to [0,1] range for numerical stability
			if (epsilon_temp > 1)
				epsilon_temp = 1;
			else if (epsilon_temp < 0)
				epsilon_temp = 0;

			epsilon[index] = epsilon_temp;
			return epsilon_temp;
		}
		else
		{
			// return (somewhat arbitrarily) a ratio of 1/2 
			epsilon[index] = 0.01f;
			return 0.01f;
		}
	}
	else	// df->type & CT_EMPTY
	{
		if (cellType[index] & cell_types::gas)
		{
			epsilon[index] = 0;
			return 0;
		}
	}
	epsilon[index] = 0;
	return 0;
}

float latticed3q19::calculateMassExchange(int currentIndex, int neighborIndex, float currentDf, float inverse_NbFi)
{
	// Table 4.1 in Nils Thuerey's PhD thesis

	if (cellType[currentIndex] & CT_NO_FLUID_NEIGH)
	{
		if (cellType[neighborIndex] & CT_NO_FLUID_NEIGH)
			return inverse_NbFi - currentDf;
		else
			// neighbor is standard cell or CT_NO_EMPTY_NEIGH
			return -currentDf;
	}
	else if (cellType[currentIndex] & CT_NO_EMPTY_NEIGH)
	{
		if (cellType[neighborIndex] & CT_NO_EMPTY_NEIGH)
			return inverse_NbFi - currentDf;
		else
			// neighbor is standard cell or CT_NO_FLUID_NEIGH
			return inverse_NbFi;
	}
	else
	{
		// current cell is standard cell
		if (cellType[neighborIndex] & CT_NO_FLUID_NEIGH)
			return inverse_NbFi;
		else if (cellType[neighborIndex] & CT_NO_EMPTY_NEIGH)
			return -currentDf;
		else
			// neighbor is standard cell
			return inverse_NbFi - currentDf;
	}
}

float3 latticed3q19::calculateNormal(int i, int j, int k)
{
	return float3
	{
		(calculateEpsilon(I3D(_width, _height, i - 1, j, k)) - calculateEpsilon(I3D(_width, _height, i + 1, j, k))) * 0.5f,
		(calculateEpsilon(I3D(_width, _height, i, j - 1, k)) - calculateEpsilon(I3D(_width, _height, i, j + 1, k))) * 0.5f,
		(calculateEpsilon(I3D(_width, _height, i, j, k - 1)) - calculateEpsilon(I3D(_width, _height, i, j, k + 1))) * 0.5f
	};
}

// This method controls the excess mass distribution. This is an implementation of part 4.3 of the reference thesis.
void latticed3q19::cellTypeAdjustment()
{
	int nb_x, nb_y, nb_z;
	int nb_cellIndex;

	// set flags for filled interface cells (interface to fluid)
	for (float3 if_to_fluid : _filledCells)
	{
		int cellIndex = I3D(_width, _height, (int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);

		//if (cellTypeTemp[cellIndex] & cell_types::CT_IF_TO_FLUID)
		{
			// convert neighboring empty cells to interface cells
			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
				nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
				nb_z = (int)(if_to_fluid.z + speedDirection[l].z);

				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

				if (cellType[nb_cellIndex] & cell_types::gas)
				{
					cellTypeTemp[nb_cellIndex] = cell_types::interphase;
					averageSurroundings(nb_x, nb_y, nb_z);
				}
				
				if (cellTypeTemp[nb_cellIndex] & cell_types::CT_IF_TO_EMPTY)
				{
					cellTypeTemp[nb_cellIndex] = cell_types::interphase;
					vector<float3>::iterator emptyIterator =
						std::find(_emptiedCells.begin(), _emptiedCells.end(), float3{ (float)nb_x, (float)nb_y, (float)nb_z });
					if (emptyIterator != _emptiedCells.end())
						_emptiedCells.erase(emptyIterator);
				}
			}
			//for (int l = 0; l < _stride; l++)
			//{
			//	nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
			//	nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
			//	nb_z = (int)(if_to_fluid.z + speedDirection[l].z);

			//	nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			//	// prevent neighboring cells from becoming empty
			//	if (cellTypeTemp[nb_cellIndex] & cell_types::CT_IF_TO_EMPTY)
			//	{
			//		cellTypeTemp[nb_cellIndex] = cell_types::interphase;
			//		vector<float3>::iterator emptyIterator = 
			//			std::find(_emptiedCells.begin(), _emptiedCells.end(), float3{ (float)nb_x, (float)nb_y, (float)nb_z });
			//		if (emptyIterator != _emptiedCells.end())
			//			_emptiedCells.erase(emptyIterator);
			//	}
			//}

			cellTypeTemp[cellIndex] = cell_types::fluid;
			cellMassTemp[cellIndex] = ro[cellIndex];
		}
	}

	//Distribute excess mass for filled cells
	for (float3 if_to_fluid : _filledCells)
	{
		float excess_mass = 0.0f;
		int cellIndex = I3D(_width, _height, (int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);
		float3 normal = calculateNormal((int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);
		float *eta = new float[19]();
		float eta_total = 0, deltaMass = 0;
		unsigned int numIF = 0;

		excess_mass = cellMass[cellIndex] - ro[cellIndex];

		// Eq. 4.9
		for (int l = 0; l < _stride; l++)
		{
			nb_x = (int)(if_to_fluid.x - speedDirection[l].x);
			nb_y = (int)(if_to_fluid.y - speedDirection[l].y);
			nb_z = (int)(if_to_fluid.z - speedDirection[l].z);

			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cellType[nb_cellIndex] & cell_types::interphase)
			{
				//eta[l] = max(0.0f, dot(normal, speedDirection[l]));
				eta[l] = dot(normal, speedDirection[l]);
				if (eta[l] <= 0)
					eta[l] = 0;
				eta_total += eta[l];
				numIF++;
			}
		}
		// If a the mass of a cell is bigger that its density, distribute excess mass along the neighboring interface cell's normal
		if (cellMass[cellIndex] > ro[cellIndex])
		if (eta_total > 0)
		{
			float eta_frac = 1 / eta_total;
			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
				nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
				nb_z = (int)(if_to_fluid.z + speedDirection[l].z);

				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

				if (cellTypeTemp[nb_cellIndex] & (cell_types::interphase))
					cellMassTemp[nb_cellIndex] += excess_mass * eta[l] * eta_frac;

				//if (cellMass[nb_cellIndex] != cellMass[nb_cellIndex])
				//	cellMass[nb_cellIndex] = cellMass[nb_cellIndex];
			}
		}
		// Evenly distribute excess mass
		//else if (numIF > 0)
		//{
		//	excess_mass = excess_mass / numIF;
		//	for (int l = 0; l < _stride; l++)
		//	{
		//		nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
		//		nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
		//		nb_z = (int)(if_to_fluid.z + speedDirection[l].z);

		//		nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
		//		
		//		cellMassTemp[nb_cellIndex] += cellTypeTemp[nb_cellIndex] & cell_types::interphase ? excess_mass : 0;

		//		//if (cellMass[nb_cellIndex] != cellMass[nb_cellIndex])
		//		//	cellMass[nb_cellIndex] = cellMass[nb_cellIndex];
		//	}
		//}

		//// after excess mass has been distributed, remaining mass equals density
		//cellTypeTemp[cellIndex] = cell_types::fluid;
		//cellMassTemp[cellIndex] = ro[cellIndex];
	}

	// set flags for emptied interface cells (interface to empty)
	for (float3 if_to_gas : _emptiedCells)
	{
		int cellIndex = I3D(_width, _height, (int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);

		//if (cellTypeTemp[cellIndex] & cell_types::CT_IF_TO_EMPTY)
		{
			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(if_to_gas.x - speedDirection[l].x);
				nb_y = (int)(if_to_gas.y - speedDirection[l].y);
				nb_z = (int)(if_to_gas.z - speedDirection[l].z);

				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

				if (cellType[nb_cellIndex] & cell_types::fluid)
				{
					cellTypeTemp[nb_cellIndex] = cell_types::interphase;
					//for (int l1 = 0; l1 < _stride; l1++)
					//	f[cellIndex* _stride + l1] = f[nb_cellIndex * _stride + l1];
				}
			}

			cellTypeTemp[cellIndex] = cell_types::gas;
			cellMassTemp[cellIndex] = 0.0f;
		}
	}

	// Distribute excess mass from empied cells
	for (float3 if_to_gas : _emptiedCells)
	{
		float excess_mass = 0.0f;
		int cellIndex = I3D(_width, _height, (int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);
		float3 normal = calculateNormal((int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);
		float eta[19] = { 0 };
		float eta_total = 0;
		unsigned int numIF = 0;

		excess_mass = cellMass[cellIndex];
		normal = float3_ScalarMultiply(-1.0f, normal);

		for (int l = 0; l < _stride; l++)
		{
			nb_x = (int)(if_to_gas.x - speedDirection[l].x);
			nb_y = (int)(if_to_gas.y - speedDirection[l].y);
			nb_z = (int)(if_to_gas.z - speedDirection[l].z);

			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cellType[nb_cellIndex] & cell_types::interphase)
			{
				eta[l] = dot(normal, speedDirection[l]);
				if (eta[l] >= 0)	eta[l] = 0;
				eta_total += eta[l];
				numIF++;
			}
		}
		// If a the mass of a cell is smaller that 0, distribute excess mass along the neighboring interface cell's normal
		if (cellMass[cellIndex]< 0)
		if (eta_total > 0)
		{
			float eta_frac = 1 / eta_total;
			for (int l = 0; l < _stride; l++)
			{
				nb_x = (int)(if_to_gas.x + speedDirection[l].x);
				nb_y = (int)(if_to_gas.y + speedDirection[l].y);
				nb_z = (int)(if_to_gas.z + speedDirection[l].z);

				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

				if (cellTypeTemp[nb_cellIndex] & cell_types::interphase)
					cellMassTemp[nb_cellIndex] += excess_mass * eta[l] * eta_frac;

				//if (cellMass[nb_cellIndex] != cellMass[nb_cellIndex])
				//	cellMass[nb_cellIndex] = cellMass[nb_cellIndex];
			}
		}
		// Evenly distribute excess mass
		//else if (numIF > 0)
		//{
		//	excess_mass = excess_mass / numIF;
		//	for (int l = 0; l < _stride; l++)
		//	{
		//		nb_x = (int)(if_to_gas.x + speedDirection[l].x);
		//		nb_y = (int)(if_to_gas.y + speedDirection[l].y);
		//		nb_z = (int)(if_to_gas.z + speedDirection[l].z);

		//		nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

		//		cellMassTemp[nb_cellIndex] += cellTypeTemp[nb_cellIndex] & cell_types::interphase ? excess_mass : 0;

		//		//if (cellMass[nb_cellIndex] != cellMass[nb_cellIndex])
		//		//	cellMass[nb_cellIndex] = cellMass[nb_cellIndex];
		//	}
		//}
		
		// after excess mass has been distributed, remaining mass equals 0
		//cellTypeTemp[cellIndex] = cell_types::gas;
		//cellMassTemp[cellIndex] = 0;
	}

	for (int i0 = 0; i0 < _numberLatticeElements; i0++)
	{
		f[i0] = ftemp[i0];
		cellMass[i0] += cellMassTemp[i0];
		cellType[i0] = cellTypeTemp[i0];

		if (cellType[i0] & (cell_types::fluid))
		{
			cellType[i0] = cellTypeTemp[i0];
			cellMass[i0] = ro[i0];
		}		

		//if (cellTypeTemp[i0] & (cell_types::interphase))
			//cellType[i0] = cellTypeTemp[i0];

	}

	_filledCells.clear();
	_emptiedCells.clear();
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
		
		if (solid[cellIndex] == 1)
			continue;

		cellType[cellIndex] |= (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);

		for (int l = 0; l < _stride; l++)
		{
			nb_x = (int)(i + speedDirection[l].x);
			nb_y = (int)(j + speedDirection[l].y);
			nb_z = (int)(k + speedDirection[l].z);

			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cellType[nb_cellIndex] & cell_types::fluid)
				cellType[cellIndex] &= ~CT_NO_FLUID_NEIGH;

			if (cellType[nb_cellIndex] & cell_types::gas)
				cellType[cellIndex] &= ~CT_NO_EMPTY_NEIGH;

			if (cellType[nb_cellIndex] & cell_types::interphase)
				cellType[cellIndex] &= ~CT_NO_IFACE_NEIGH;
		}

		if (cellType[cellIndex] & CT_NO_EMPTY_NEIGH)
			cellType[cellIndex] &= ~CT_NO_FLUID_NEIGH;
	}
}


void latticed3q19::averageSurroundings(int i, int j, int k)
{
	int n = 0, newI= 0, newJ = 0, newK= 0;	// counter
	float3 nb_cell = { 0, 0, 0 };
	int cellIndex = I3D(_width, _height, i, j, k), neighborIndex =0;
	float rho = 0.0f;
	float3 velocity{ 0, 0, 0 };
	// set mass initially to zero
	//cellMass[cellIndex] = 0;
	
	// average density and velocity of surrounding cells
	for (int l = 0; l < _stride; l++)		
	{
		newI = (int)(i + speedDirection[l].x);
		newJ = (int)(j + speedDirection[l].y);
		newK = (int)(k + speedDirection[l].z);

		neighborIndex = I3D(_width, _height, newI, newJ, newK);
		
		// fluid or interface cells only
		if (cellType[neighborIndex] & ( cell_types::interphase)) //cell_types::fluid |
		{
			rho += ro[neighborIndex];
			velocity.x += velocityVector[neighborIndex].x;
			velocity.y += velocityVector[neighborIndex].y;
			velocity.z += velocityVector[neighborIndex].z;
			
			n++;
		}
	}
	if (n > 0)
	{
		rho /= n;
		velocity.x /= n;
		velocity.y /= n;
		velocity.z /= n;

		calculateEquilibriumFunction(velocity, rho);

		for (int l = 0; l < _stride; l++)
		{
			ftemp[cellIndex * _stride + l] = feq[l];
			//ftemp[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
			//	feq[l] + feq[inverseSpeedDirectionIndex[l]] - f[cellIndex * _stride + l];
		}
	}
}

void latticed3q19::setFilledOrEmpty(int i, int j, int k)
{
	int cellIndex = I3D(_width, _height, i, j, k);
	
	//Check interfase cell mass for flag reinitialization
	if (cellType[cellIndex] & cell_types::interphase)
	{
		if (
			(cellMass[cellIndex] >((1 + FILL_OFFSET) * ro[cellIndex])) // Eq. (4.7)
			|| // Remove artifacts
			((cellMass[cellIndex] > ((1 - LONELY_THRESH)* ro[cellIndex])) && (cellType[cellIndex] & CT_NO_FLUID_NEIGH))
			)
		{
			// interface to fluid cell
			cellTypeTemp[cellIndex] = cell_types::CT_IF_TO_FLUID;
			_filledCells.push_back(float3{ (float)i, (float)j, (float)k });
		}
		else if (
			(cellMass[cellIndex] < -FILL_OFFSET * ro[cellIndex]) // Eq. (4.7)
			||	
			((cellMass[cellIndex] <= LONELY_THRESH*ro[cellIndex]) && (cellType[cellIndex] & CT_NO_FLUID_NEIGH))
			)	// isolated interface cell: only empty or obstacle neighbors
		{
			// interface to empty cell
			cellTypeTemp[cellIndex] = cell_types::CT_IF_TO_EMPTY;
			_emptiedCells.push_back(float3{ (float)i, (float)j, (float)k });
		}
	}

	// clear neighborhood flags (will be determined later)
	cellTypeTemp[cellIndex] &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
}

// ////Working code without free surfaces - lattice with fluid and gravity
//#include "Lattice.h"
//
//latticed3q19::latticed3q19(int width, int height, int depth, float worldViscosity, float mass, float cellsPerSide, float domainSize)
//{
//	_width = width; _height = height; _depth = depth;
//	_stride = 19;
//	_numberLatticeElements = _width * _height * _depth;
//	_numberAllElements = _stride * _numberLatticeElements;
//	_cellsPerSide = cellsPerSide;
//	_domainSize = domainSize;
//	_mass = mass;
//
//	f = new float[_numberAllElements]();
//	ftemp = new float[_numberAllElements]();
//	feq = new float[_stride]();
//	solid = new unsigned int[_numberLatticeElements]();
//	velocityVector = new float3[_numberLatticeElements]();
//	ro = new float[_numberLatticeElements]();
//
//	epsilon = new float[_numberLatticeElements]();
//	cellType = new int[_numberLatticeElements];
//	cellTypeTemp = new int[_numberLatticeElements];
//	cellMass = new float[_numberLatticeElements]();
//	cellMassTemp = new float[_numberLatticeElements]();
//
//	for (int i = 0; i < _numberLatticeElements; i++)
//		cellType[i] = cellTypeTemp[i] = gas;
//
//	c = (float)(1.0 / sqrt(3.0));
//
//	cellSize = _domainSize / _cellsPerSide;
//
//	gravity = -9.8f;
//
//	timeStep = (float)(sqrtf((0.005f * cellSize) / fabs(gravity)));
//
//	_vMax = cellSize / timeStep;
//
//	viscosity = worldViscosity * timeStep / (cellSize * cellSize);
//
//	_tau = 3.0f * viscosity + 0.5f;
//
//	_w = 1 / _tau;
//
//	latticeAcceleration = gravity * timeStep * timeStep / cellSize;
//}
//
//latticed3q19::~latticed3q19()
//{
//	delete[] f;
//	delete[] ftemp;
//	delete[] feq;
//	delete[] solid;
//	delete[] ro;
//	delete[] velocityVector;
//	delete[] epsilon;
//	delete[] cellType;
//	delete[] cellMass;
//	delete[] cellMassTemp;
//}
//
//void latticed3q19::step(void)
//{
//	//setNeighborhoodFlags();
//
//	stream();
//
//	collide();
//
//	//cellTypeAdjustment();
//
//	//applyBoundaryConditions();
//}
//
//void latticed3q19::stream()
//{
//	int cellIndex, cellDf, advectedDf, advectedCell, inverseAdvectedDf;
//	int newI, newJ, newK;
//	float3 normal;
//
//	for (int k = 0; k<_depth; k++)
//	{
//		for (int j = 0; j < _height; j++)
//		{
//			for (int i = 0; i<_width; i++)
//			{
//				cellIndex = I3D(_width, _height, i, j, k);
//
//				if (solid[cellIndex] == 0)
//				{
//					if (cellType[cellIndex] & cell_types::fluid)
//					{
//						try
//						{
//							//if (cellMass[cellIndex] == ro[cellIndex])
//							{
//								cellMassTemp[cellIndex] = 0;
//
//								for (int l = 0; l < 19; l++)
//								{
//									newI = (int)(i + speedDirection[l].x);
//									newJ = (int)(j + speedDirection[l].y);
//									newK = (int)(k + speedDirection[l].z);
//
//									//newI = (int)(i + speedDirection[inverseSpeedDirectionIndex[l]].x);
//									//newJ = (int)(j + speedDirection[inverseSpeedDirectionIndex[l]].y);
//									//newK = (int)(k + speedDirection[inverseSpeedDirectionIndex[l]].z);
//
//									cellDf = cellIndex*_stride + l;
//									advectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, l);
//									advectedCell = I3D(_width, _height, newI, newJ, newK);
//
//									//if (cellType[advectedCell] & (cell_types::fluid | cell_types::interphase))
//									//{
//									//	//newI = (int)(i + speedDirection[inverseSpeedDirectionIndex[l]].x);
//									//	//newJ = (int)(j + speedDirection[inverseSpeedDirectionIndex[l]].y);
//									//	//newK = (int)(k + speedDirection[inverseSpeedDirectionIndex[l]].z);
//									//	inverseAdvectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, inverseSpeedDirectionIndex[l]);
//
//									//	// mass exchange between fluid and interface cell, Eq. (4.2)
//									//	cellMassTemp[cellIndex] += f[inverseAdvectedDf] - f[cellDf];
//									//}								
//									ftemp[cellDf] = f[advectedDf];
//								}
//							}
//							//else
//							//	throw std::string("Not the same mass (" +std::to_string(cellMass[cellIndex]) 
//							//	+ ") and density (" + std::to_string(ro[cellIndex]) + ") at ") + std::to_string(cellIndex);
//						}
//						catch (std::string message) { cout << "Exception: " << message.c_str() << endl; }
//						catch (...){ cout << "Default exception." << endl; }
//					}
//					//else if (cellType[cellIndex] & cell_types::interphase)
//					//{
//					//	float currentEpsilon = calculateEpsilon(cellIndex);
//					//	
//					//	cellMassTemp[cellIndex] = 0;
//					//	// Calculate air equilibrium function to reconstruct missing dfs
//					//	calculateEquilibriumFunction(velocityVector[cellIndex], 1.0f);
//
//					//	for (int l = 0; l < _stride; l++)
//					//	{
//					//		newI = (int)(i + speedDirection[l].x);
//					//		newJ = (int)(j + speedDirection[l].y);
//					//		newK = (int)(k + speedDirection[l].z);
//					//		
//					//		cellDf = cellIndex*_stride + l;
//					//		advectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, l);
//					//		advectedCell = I3D(_width, _height, newI, newJ, newK);
//					//		inverseAdvectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, inverseSpeedDirectionIndex[l]);
//					//		normal = calculateNormal(i, j, k);
//
//					//		//if (cellType[advectedCell] & cell_types::fluid)
//					//		//{
//					//		//	// mass exchange between fluid and interface cell, Eq. (4.2)
//					//		//	//ftemp[cellDf] = f[advectedDf];
//					//		//}
//					//		if (cellType[advectedCell] & cell_types::interphase)
//					//		{
//					//			// mass exchange between two interface cells, Eq. (4.3)
//					//			float neighborEpsilon = calculateEpsilon(advectedCell);
//					//			float massExchange = calculateMassExchange(cellIndex, advectedCell, f[cellDf], f[inverseAdvectedDf]);
//
//					//			cellMassTemp[cellIndex] += (f[inverseAdvectedDf] - f[cellDf]) * 0.5f * (currentEpsilon + neighborEpsilon);
//					//			//cellMassTemp[cellIndex] += (massExchange)*0.5f * (currentEpsilon + neighborEpsilon);
//
//					//			ftemp[cellDf] = f[advectedDf];
//					//		}
//					//		else if (cellType[advectedCell] & cell_types::gas) 	// Eq. (4.6)
//					//		{
//					//			ftemp[cellDf] = feq[l] + feq[inverseSpeedDirectionIndex[l]] - f[cellDf];
//					//		}
//					//	}
//
//					//	//// always use reconstructed atmospheric distribution function for directions along surface normal;
//					//	//// separate loop to handle mass exchange correctly
//					//	for (int l = 0; l < 19; l++)		
//					//	{
//					//		if (dot(normal, speedDirection[inverseSpeedDirectionIndex[i]]) > 0)		// Eq. (4.6)
//					//		// reconstructed atmospheric distribution function, Eq. (4.5)
//					//			ftemp[cellDf] = feq[l] + feq[inverseSpeedDirectionIndex[l]] - f[cellDf];							
//					//	}
//					//}
//				}
//				else
//				{
//					for (int l = 0; l < 19; l++)
//					{
//						cellDf = cellIndex*_stride + l;
//						advectedDf = cellIndex*_stride + inverseSpeedDirectionIndex[l];
//
//						ftemp[cellDf] = f[advectedDf];
//					}
//				}
//			}
//		}
//	}
//
//	for (int i0 = 0; i0 < _numberAllElements; i0++)
//	{
//		f[i0] = ftemp[i0];
//		//if (i0 % _stride == 0)
//		//{
//		//	int index = i0 / _stride;
//
//		//	if (cellType[index] & (cell_types::fluid))// | cell_types::interphase))
//		//	{
//		//		cellMass[index] += cellMassTemp[index];
//		//		//epsilon[index] = cellMass[index] / ro[index];
//		//	}
//
//		//	//if (cellType[index] & (cell_types::fluid))
//		//	//{
//		//	//	cellMass[index] = ro[index];
//		//	//	epsilon[index] = 1.0f;
//		//	//}
//		//}
//	}
//
//}
//
//void latticed3q19::collide(void)
//{
//	int iBase = 0, numFluid = 0;
//	float roAverage = 0;
//	//for (int i0 = 0; i0 < _numberLatticeElements; i0++)
//	for (int k = 0; k<_depth; k++)
//	for (int j = 0; j < _height; j++)
//	for (int i = 0; i<_width; i++)
//	{
//		int i0 = I3D(_width, _height, i, j, k);
//		if (solid[i0] == 0)
//		{
//			if (cellType[i0] & (cell_types::fluid))// | cell_types::interphase))
//			{
//				deriveQuantities(i0);
//				calculateEquilibriumFunction(velocityVector[i0], ro[i0]);;
//
//				for (int l = 0; l < 19; l++)
//				{
//					iBase = i0 * _stride + l;
//					//f[iBase] = f[iBase] - (f[iBase] - feq[l]) / _tau;
//					f[iBase] = (1.0f - _w) * f[iBase] + _w * feq[l]
//						//To include gravity
//						+ latticeWeights[l] * ro[i0] * dot(speedDirection[l], float3{ 0, latticeAcceleration, 0 });
//
//					if (f[iBase] > 1 || f[iBase] < 0)
//						cout << "error" << endl;
//				}
//
//				epsilon[i0] = cellMass[i0] / ro[i0];
//
//				roAverage += ro[i0];
//				numFluid++;
//				//setFilledOrEmpty(i, j, k);
//
//				//if (cellMassTemp[i0] < 0.0f)
//				//if (cellType[i0] & cell_types::fluid)
//				//	cellMass[i0] = ro[i0];
//				//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
//			}
//		}
//		//else
//		//	solid_BC(i0);
//	}
//	cout << "RO: " << roAverage / numFluid << endl;
//}
//
//void latticed3q19::applyBoundaryConditions()
//{
//	//in_BC(vector3d(0.0,0.0, -0.6));
//}
//
//void latticed3q19::solid_BC(int i0)
//{
//	float temp;
//
//	temp = f[i0*_stride + 1]; 	f[i0*_stride + 1] = f[i0*_stride + 2];		f[i0*_stride + 2] = temp;		// f1	<-> f2
//	temp = f[i0*_stride + 3];	f[i0*_stride + 3] = f[i0*_stride + 4];		f[i0*_stride + 4] = temp;		// f3	<-> f4
//	temp = f[i0*_stride + 5];	f[i0*_stride + 5] = f[i0*_stride + 6];		f[i0*_stride + 6] = temp;		// f5	<-> f6
//	temp = f[i0*_stride + 7];	f[i0*_stride + 7] = f[i0*_stride + 12];		f[i0*_stride + 12] = temp;		// f7	<-> f12
//	temp = f[i0*_stride + 8];	f[i0*_stride + 8] = f[i0*_stride + 11];		f[i0*_stride + 11] = temp;		// f8	<-> f11
//	temp = f[i0*_stride + 9];	f[i0*_stride + 9] = f[i0*_stride + 14];		f[i0*_stride + 14] = temp;		// f9	<-> f14
//	temp = f[i0*_stride + 10];	f[i0*_stride + 10] = f[i0*_stride + 13];	f[i0*_stride + 13] = temp;		// f10	<-> f13
//	temp = f[i0*_stride + 15];	f[i0*_stride + 15] = f[i0*_stride + 18];	f[i0*_stride + 18] = temp;		// f15	<-> f18
//	temp = f[i0*_stride + 16];	f[i0*_stride + 16] = f[i0*_stride + 17];	f[i0*_stride + 17] = temp;		// f16	<-> f17
//}
//
//void latticed3q19::calculateSpeedVector(int index)
//{
//	//calculateRo();
//	//rovx = rovy = rovz = 0; 
//
//	ro[index] = rovx = rovy = rovz = 0;
//	int i0 = 0;
//	for (int i = 0; i<_stride; i++)
//	{
//		i0 = index * _stride + i;
//		ro[index] += f[i0];
//		rovx += f[i0] * speedDirection[i].x;
//		rovy += f[i0] * speedDirection[i].y;
//		rovz += f[i0] * speedDirection[i].z;
//	}
//
//	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
//	if (ro[index] == ro[index] && ro[index] != 0.0f)
//	{
//		velocityVector[index].x = rovx / ro[index];
//		velocityVector[index].y = rovy / ro[index];
//		velocityVector[index].z = rovz / ro[index];
//	}
//	else
//	{
//		velocityVector[index].x = 0;
//		velocityVector[index].y = 0;
//		velocityVector[index].z = 0;
//	}
//}
//
//void latticed3q19::calculateEquilibriumFunction(float3 inVector, float inRo)
//{
//	float w;
//	float eiU = 0;	// Dot product between speed direction and velocity
//	float eiUsq = 0; // Dot product squared
//	float uSq = dot(inVector, inVector);	//Velocity squared
//
//	for (int i = 0; i<_stride; i++)
//	{
//		w = latticeWeights[i];
//		eiU = dot(speedDirection[i], inVector);
//		eiUsq = eiU * eiU;
//		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
//		feq[i] = w * inRo * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
//	}
//}
//
//void latticed3q19::calculateInEquilibriumFunction(int index, float3 inVector, float inRo)
//{
//	float w;
//	float eiU = 0;	// Dot product between speed direction and velocity
//	float eiUsq = 0; // Dot product squared
//	float uSq = 0;	//Velocity squared
//
//	uSq = dot(inVector, inVector);
//	int iBase = 0;
//	// For the interfase cells, an atmospheric pressure (1.0) is used as density
//	ro[index] = inRo;
//	//cellType[index] & (cell_types::fluid) ? inRo :
//	//cellType[index] & (cell_types::interphase) ? FILL_OFFSET * inRo : 1.0f;
//
//	if (cellType[index] & (cell_types::fluid | cell_types::interphase))
//	for (int i = 0; i<_stride; i++)
//	{
//		w = latticeWeights[i];
//		eiU = dot(speedDirection[i], inVector);
//		eiUsq = eiU * eiU;
//
//		iBase = index*_stride + i;
//		//ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
//		ftemp[iBase] = f[iBase] = w;// *ro[index] * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
//	}
//}
//
//void latticed3q19::deriveQuantities(int index)
//{
//	int dfIndex = 0;
//
//	// Calculate average density
//	ro[index] = 0;
//	for (int l = 0; l < _stride; l++)
//		ro[index] += f[index * _stride + l];
//
//	velocityVector[index] = float3{ 0, 0, 0 };
//
//	if (ro[index] >= 0)
//	{
//		for (int l = 0; l < _stride; l++)
//		{
//			dfIndex = index * _stride + l;
//			velocityVector[index].x += f[dfIndex] * speedDirection[l].x;
//			velocityVector[index].y += f[dfIndex] * speedDirection[l].y;
//			velocityVector[index].z += f[dfIndex] * speedDirection[l].z;
//		}
//		velocityVector[index].x /= ro[index];
//		velocityVector[index].y /= ro[index];
//		velocityVector[index].z /= ro[index];
//	}
//
//	float n = float3_Norm(velocityVector[index]);
//	if (n > _vMax)
//		velocityVector[index] = float3_ScalarMultiply(_vMax / n, velocityVector[index]);
//}
//
//void latticed3q19::calculateInitialMass()
//{
//	float individual_cell_mass = 0.0f, fluidCells = 0;
//
//	for (int i = 0; i < _numberLatticeElements; i++)
//	{
//		if (cellType[i] & cell_types::fluid)
//			fluidCells++;
//	}
//
//	individual_cell_mass = _mass / fluidCells;
//
//	for (int i = 0; i < _numberLatticeElements; i++)
//	{
//		if (cellType[i] & cell_types::fluid)
//		{
//			cellMass[i] = cellMassTemp[i] = ro[i] = individual_cell_mass;
//			epsilon[i] = cellMass[i] / ro[i];
//		}
//		else if (cellType[i] & cell_types::interphase)
//		{
//			cellMass[i] = cellMassTemp[i] = FILL_OFFSET * ro[i];
//			epsilon[i] = cellMass[i] / ro[i];
//		}
//	}
//}
//
//float latticed3q19::calculateEpsilon(int index)
//{
//	if ((cellType[index] & cell_types::fluid) || (solid[index] == 1))
//	{
//		epsilon[index] = 1;
//		return 1;
//	}
//	else if (cellType[index] & cell_types::interphase)
//	{
//		if (ro[index] > 0)
//		{
//			float epsilon_temp = cellMass[index] / ro[index];
//
//			// df->mass can even be < 0 or > df->rho for interface cells to be converted to fluid or empty cells in the next step;
//			// clamp to [0,1] range for numerical stability
//			if (epsilon_temp > 1)
//				epsilon_temp = 1;
//			else if (epsilon_temp < 0)
//				epsilon_temp = 0;
//
//			epsilon[index] = epsilon_temp;
//			return epsilon_temp;
//		}
//		else
//		{
//			// return (somewhat arbitrarily) a ratio of 1/2 
//			epsilon[index] = 0.5f;
//			return 0.5f;
//		}
//	}
//	else	// df->type & CT_EMPTY
//	{
//		if (cellType[index] & cell_types::gas)
//		{
//			epsilon[index] = 0;
//			return 0;
//		}
//	}
//	epsilon[index] = 0;
//	return 0;
//}
//
//float latticed3q19::calculateMassExchange(int currentIndex, int neighborIndex, float currentDf, float inverse_NbFi)
//{
//	// Table 4.1 in Nils Thuerey's PhD thesis
//
//	if (cellType[currentIndex] & CT_NO_FLUID_NEIGH)
//	{
//		if (cellType[neighborIndex] & CT_NO_FLUID_NEIGH)
//			return inverse_NbFi - currentDf;
//		else
//			// neighbor is standard cell or CT_NO_EMPTY_NEIGH
//			return -currentDf;
//	}
//	else if (cellType[currentIndex] & CT_NO_EMPTY_NEIGH)
//	{
//		if (cellType[neighborIndex] & CT_NO_EMPTY_NEIGH)
//			return inverse_NbFi - currentDf;
//		else
//			// neighbor is standard cell or CT_NO_FLUID_NEIGH
//			return inverse_NbFi;
//	}
//	else
//	{
//		// current cell is standard cell
//		if (cellType[neighborIndex] & CT_NO_FLUID_NEIGH)
//			return -currentDf;
//		else if (cellType[neighborIndex] & CT_NO_EMPTY_NEIGH)
//			return inverse_NbFi;
//		else
//			// neighbor is standard cell
//			return inverse_NbFi - currentDf;
//	}
//}
//
//float3 latticed3q19::calculateNormal(int i, int j, int k)
//{
//	return float3
//	{
//	(calculateEpsilon(I3D(_width, _height, i - 1, j, k)) - calculateEpsilon(I3D(_width, _height, i + 1, j, k))) * 0.5f,
//	(calculateEpsilon(I3D(_width, _height, i, j - 1, k)) - calculateEpsilon(I3D(_width, _height, i, j + 1, k))) * 0.5f,
//	(calculateEpsilon(I3D(_width, _height, i, j, k - 1)) - calculateEpsilon(I3D(_width, _height, i, j, k + 1))) * 0.5f
//};
//}
//
//void latticed3q19::cellTypeAdjustment()
//{
//	int nb_x, nb_y, nb_z;
//	int nb_cellIndex;
//
//	// set flags for filled interface cells (interface to fluid)
//	for (float3 if_to_fluid : _filledCells)
//	{
//		int cellIndex = I3D(_width, _height, (int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);
//
//		if (cellType[cellIndex] & cell_types::CT_IF_TO_FLUID)
//		{
//			// convert neighboring empty cells to interface cells
//			for (int l = 0; l < _stride; l++)
//			{
//				nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
//				nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
//				nb_z = (int)(if_to_fluid.z + speedDirection[l].z);
//
//				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//				if (cellType[nb_cellIndex] & cell_types::gas)
//				{
//					cellType[nb_cellIndex] = cell_types::interphase;
//					averageSurroundings(nb_x, nb_y, nb_z);
//				}
//
//				// prevent neighboring cells from becoming empty
//				if (cellType[nb_cellIndex] & cell_types::CT_IF_TO_EMPTY)
//				{
//					cellType[nb_cellIndex] = cell_types::interphase;
//					_emptiedCells.erase(remove(_emptiedCells.begin(), _emptiedCells.end(), float3{ (float)nb_x, (float)nb_y, (float)nb_z }));
//				}
//			}
//
//			cellType[cellIndex] = cell_types::fluid;
//		}
//	}
//
//	// set flags for emptied interface cells (interface to empty)
//	for (float3 if_to_gas : _emptiedCells)
//	{
//		int cellIndex = I3D(_width, _height, (int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);
//
//		if (cellType[cellIndex] & cell_types::CT_IF_TO_EMPTY)
//		{
//			for (int l = 0; l < _stride; l++)
//			{
//				nb_x = (int)(if_to_gas.x + speedDirection[l].x);
//				nb_y = (int)(if_to_gas.y + speedDirection[l].y);
//				nb_z = (int)(if_to_gas.z + speedDirection[l].z);
//
//				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//				if (cellType[nb_cellIndex] & cell_types::fluid)
//				{
//					cellType[nb_cellIndex] = cell_types::interphase;
//					//for (int l1 = 0; l1 < _stride; l1++)
//					//	f[nb_cellIndex* _stride + l1] = f[cellIndex * _stride + l1];
//				}
//			}
//
//			cellType[cellIndex] = cell_types::gas;
//		}
//	}
//
//	//Distribute excess mass for filled cells
//	for (float3 if_to_fluid : _filledCells)
//	{
//		float excess_mass = 0.0f;
//		int cellIndex = I3D(_width, _height, (int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);
//		float3 normal = calculateNormal((int)if_to_fluid.x, (int)if_to_fluid.y, (int)if_to_fluid.z);
//		float *eta = new float[19]();
//		float eta_total = 0;
//		unsigned int numIF = 0;
//
//		excess_mass = cellMass[cellIndex] - ro[cellIndex];
//
//		for (int l = 0; l < _stride; l++)
//		{
//			nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
//			nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
//			nb_z = (int)(if_to_fluid.z + speedDirection[l].z);
//
//			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//			//if (cellType[nb_cellIndex] & cell_types::interphase)
//			{
//				eta[l] = dot(normal, speedDirection[l]);
//				if (eta[l] <= 0)	eta[l] = 0;
//				eta_total += eta[l];
//				numIF++;
//			}
//		}
//		if (cellMass[cellIndex] > ro[cellIndex])
//			//if (eta_total > 0)
//			//{
//			//	float eta_frac = 1 / eta_total;
//			//	for (int l = 0; l < _stride; l++)
//			//	{
//			//		nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
//			//		nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
//			//		nb_z = (int)(if_to_fluid.z + speedDirection[l].z);
//
//			//		nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//			//		if (cellType[nb_cellIndex] & cell_types::interphase)
//			//			cellMass[nb_cellIndex] += excess_mass * eta[l] * eta_frac;
//			//	}
//			//}
//			//else if (numIF > 0)
//		{
//			excess_mass = excess_mass / numIF;
//			for (int l = 0; l < _stride; l++)
//			{
//				nb_x = (int)(if_to_fluid.x + speedDirection[l].x);
//				nb_y = (int)(if_to_fluid.y + speedDirection[l].y);
//				nb_z = (int)(if_to_fluid.z + speedDirection[l].z);
//
//				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//				cellMass[nb_cellIndex] += cellType[nb_cellIndex] & cell_types::interphase ? excess_mass : 0;
//			}
//		}
//
//		// after excess mass has been distributed, remaining mass equals density
//		cellMass[cellIndex] = ro[cellIndex];
//	}
//
//	for (float3 if_to_gas : _emptiedCells)
//	{
//		float excess_mass = 0.0f;
//		int cellIndex = I3D(_width, _height, (int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);
//		float3 normal = calculateNormal((int)if_to_gas.x, (int)if_to_gas.y, (int)if_to_gas.z);
//		float eta[19] = { 0 };
//		float eta_total = 0;
//		unsigned int numIF = 0;
//
//		excess_mass = -cellMass[cellIndex];
//		float3_ScalarMultiply(-1.0f, normal);
//
//		for (int l = 0; l < _stride; l++)
//		{
//			nb_x = (int)(if_to_gas.x + speedDirection[l].x);
//			nb_y = (int)(if_to_gas.y + speedDirection[l].y);
//			nb_z = (int)(if_to_gas.z + speedDirection[l].z);
//
//			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//			//if (cellType[nb_cellIndex] & cell_types::interphase)
//			{
//				eta[l] = dot(normal, speedDirection[l]);
//				if (eta[l] < 0)	eta[l] = 0;
//				eta_total += eta[l];
//				numIF++;
//			}
//		}
//		if (cellMass[cellIndex]< 0)
//			//if (eta_total > 0)
//			//{
//			//	float eta_frac = 1 / eta_total;
//			//	for (int l = 0; l < _stride; l++)
//			//	{
//			//		nb_x = (int)(if_to_gas.x + speedDirection[l].x);
//			//		nb_y = (int)(if_to_gas.y + speedDirection[l].y);
//			//		nb_z = (int)(if_to_gas.z + speedDirection[l].z);
//
//			//		nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//			//		if (cellType[nb_cellIndex] & cell_types::interphase)
//			//			cellMass[nb_cellIndex] += excess_mass * eta[l] * eta_frac;
//			//	}
//			//}
//			//else if (numIF > 0)
//		{
//			excess_mass = excess_mass / numIF;
//			for (int l = 0; l < _stride; l++)
//			{
//				nb_x = (int)(if_to_gas.x + speedDirection[l].x);
//				nb_y = (int)(if_to_gas.y + speedDirection[l].y);
//				nb_z = (int)(if_to_gas.z + speedDirection[l].z);
//
//				nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//				cellMass[nb_cellIndex] += cellType[nb_cellIndex] & cell_types::interphase ? excess_mass : 0;
//			}
//		}
//
//		// after excess mass has been distributed, remaining mass equals 0
//		cellMass[cellIndex] = 0;
//	}
//
//	_filledCells.clear();
//	_emptiedCells.clear();
//}
//
//void latticed3q19::setNeighborhoodFlags()
//{
//	int nb_x, nb_y, nb_z;
//	int nb_cellIndex;
//
//	for (int k = 0; k<_depth; k++)
//	for (int j = 0; j < _height; j++)
//	for (int i = 0; i < _width; i++)
//	{
//		int cellIndex = I3D(_width, _height, i, j, k);
//
//		if (solid[cellIndex] == 1)
//			continue;
//
//		cellType[cellIndex] |= (CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
//
//		for (int l = 0; l < _stride; l++)
//		{
//			nb_x = (int)(i + speedDirection[l].x);
//			nb_y = (int)(j + speedDirection[l].y);
//			nb_z = (int)(k + speedDirection[l].z);
//
//			nb_cellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
//
//			if (cellType[nb_cellIndex] & cell_types::fluid)
//				cellType[cellIndex] &= ~CT_NO_FLUID_NEIGH;
//
//			if (cellType[nb_cellIndex] & cell_types::gas)
//				cellType[cellIndex] &= ~CT_NO_EMPTY_NEIGH;
//
//			if (cellType[nb_cellIndex] & cell_types::interphase)
//				cellType[cellIndex] &= ~CT_NO_IFACE_NEIGH;
//		}
//
//		if (cellType[cellIndex] & CT_NO_EMPTY_NEIGH)
//			cellType[cellIndex] &= ~CT_NO_FLUID_NEIGH;
//	}
//}
//
//
//void latticed3q19::averageSurroundings(int i, int j, int k)
//{
//	int n = 0, newI = 0, newJ = 0, newK = 0;	// counter
//	float3 nb_cell = { 0, 0, 0 };
//	int cellIndex = I3D(_width, _height, i, j, k), neighborIndex = 0;
//	// set mass initially to zero
//	//cellMass[cellIndex] = 0;
//
//	// average density and velocity of surrounding cells
//	ro[cellIndex] = 0;
//	velocityVector[cellIndex] = float3{ 0, 0, 0 };
//
//	for (int l = 0; l < _stride; l++)
//	{
//		newI = (int)(i + speedDirection[l].x);
//		newJ = (int)(j + speedDirection[l].y);
//		newK = (int)(k + speedDirection[l].z);
//
//		neighborIndex = I3D(_width, _height, newI, newJ, newK);
//
//		// fluid or interface cells only
//		if (cellType[neighborIndex] & (cell_types::fluid | cell_types::interphase))
//		{
//			ro[cellIndex] += ro[neighborIndex];
//			velocityVector[cellIndex].x += velocityVector[neighborIndex].x;
//			velocityVector[cellIndex].y += velocityVector[neighborIndex].y;
//			velocityVector[cellIndex].z += velocityVector[neighborIndex].z;
//
//			n++;
//		}
//	}
//	if (n > 0)
//	{
//		ro[cellIndex] /= n;
//		velocityVector[cellIndex].x /= n;
//		velocityVector[cellIndex].y /= n;
//		velocityVector[cellIndex].z /= n;
//	}
//
//	//applyGravity(cellIndex);
//
//	// calculate equilibrium distribution function
//	calculateEquilibriumFunction(velocityVector[cellIndex], ro[cellIndex]);
//
//	for (int l = 0; l < _stride; l++)
//		f[cellIndex * _stride + l] = feq[l];
//}
//
//void latticed3q19::setFilledOrEmpty(int i, int j, int k)
//{
//	int cellIndex = I3D(_width, _height, i, j, k);
//	//Check interfase cell mass for flag reinitialization
//	if (cellType[cellIndex] & cell_types::interphase)
//	{
//		if (
//			(cellMass[cellIndex] >((1 + FILL_OFFSET) * ro[cellIndex]))
//			||
//			((cellMass[cellIndex] > ((1 - LONELY_THRESH)* ro[cellIndex])) && (cellType[cellIndex] & CT_NO_FLUID_NEIGH))
//			)
//		{
//			// interface to fluid cell
//			cellType[cellIndex] = cell_types::CT_IF_TO_FLUID;
//			_filledCells.push_back(float3{ (float)i, (float)j, (float)k });
//		}
//		else if (
//			(cellMass[cellIndex] < -FILL_OFFSET * ro[cellIndex])
//			||
//			((cellMass[cellIndex] <= LONELY_THRESH*ro[cellIndex]) && (cellType[cellIndex] & CT_NO_FLUID_NEIGH))
//			)	// isolated interface cell: only empty or obstacle neighbors
//		{
//			// interface to empty cell
//			cellType[cellIndex] = cell_types::CT_IF_TO_EMPTY;
//			_emptiedCells.push_back(float3{ (float)i, (float)j, (float)k });
//		}
//	}
//
//	// clear neighborhood flags (will be determined later)
//	cellType[cellIndex] &= ~(CT_NO_FLUID_NEIGH | CT_NO_EMPTY_NEIGH | CT_NO_IFACE_NEIGH);
//}
