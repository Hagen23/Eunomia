#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float worldViscosity, float cellsPerSide, float domainSize)
{
	_width = width; _height = height; _depth = depth;
	_stride = 19;
	_numberLatticeElements = _width * _height * _depth;
	_numberAllElements = _stride * _numberLatticeElements;
	_cellsPerSide = cellsPerSide;
	_domainSize = domainSize;

	f = new float[_numberAllElements]();
	ftemp = new float[_numberAllElements]();
	velocityVector = new float3[_numberLatticeElements]();
	ro = new float[_numberLatticeElements]();

	cell_mass = new float[_numberLatticeElements]();
	cell_mass_temp = new float[_numberLatticeElements]();

	cell_type = new cell_types[_numberLatticeElements]();
	cell_type_temp = new cell_types[_numberLatticeElements]();

	nFluidNbs = new int[_numberLatticeElements]();
	nGasNbs = new int[_numberLatticeElements]();
	nInterNbs = new int[_numberLatticeElements]();
	nSolidNbs = new int[_numberLatticeElements]();
	
	c = (float)(1.0 / sqrt(3.0));
	
	// These values are needed to maintain the fluid stability. Part 3.3 of the reference Thesis
	cellSize = _domainSize / _cellsPerSide;

	gravity = 9.8f;

	timeStep = (float)(sqrtf( (0.005f * cellSize) / fabs(gravity)));

	_vMax = cellSize / timeStep;

	viscosity = worldViscosity * timeStep / (cellSize * cellSize);

	_tau = 3.0f * viscosity + 0.5f;

	_w = 1.0f / _tau;

	latticeAcceleration = gravity * timeStep * timeStep / cellSize;

	//initLatticeDistributions();
}
	
latticed3q19::~latticed3q19()
{
	delete[] f;
	delete[] ftemp;
	delete[] ro;
	delete[] velocityVector;
}

void latticed3q19::step(void)
{
	//setNumberNeighbors();

	//calculateMassExchange();

	stream();
	
	reconstructInterfaceDfs();

	collide();

	//flagReinitialization();

	//distributeExcessMass();

	//applyBoundaryConditions();
}

void latticed3q19::stream()
{
	int cellIndex, advectedCellIndex, cellDf, advectedDf, inverseAdvectedDf;
	int newI, newJ, newK;
	float deltaMass = 0;

	for (int k = 1; k<_depth-1; k++)
	for (int j = 1; j < _height-1; j++)
	for (int i = 1; i<_width-1; i++)
	{
		cellIndex = I3D(_width, _height, i, j, k);

		if (cell_type[cellIndex] != cell_types::solid && cell_type[cellIndex] != cell_types::gas)
		{
			deltaMass = 0;
			//calculateEquilibriumFunction(velocityVector[cellIndex], 1.0f);
			float3 cell_normal = calculateNormal(i, j, k);

			for (int l = 0; l < 19; l++)
			{
				newI = (int)(i + inverseSpeedDirection[l].x);
				newJ = (int)(j + inverseSpeedDirection[l].y);
				newK = (int)(k + inverseSpeedDirection[l].z);

				cellDf = cellIndex * _stride + l;
				advectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, l);
				inverseAdvectedDf = I3D_S(_width, _height, _stride, newI, newJ, newK, inverseSpeedDirectionIndex[l]);
				advectedCellIndex = I3D(_width, _height, newI, newJ, newK);

				if (cell_type[cellIndex] == cell_types::fluid || cell_type[cellIndex] == cell_types::interfase)
				{
					ftemp[cellDf] = f[advectedDf];
				}
				//else if (cell_type[cellIndex] == cell_types::interfase)
				//{
				//	if (cell_type[advectedCellIndex] == cell_types::fluid)
				//	{
				//		deltaMass += f[inverseAdvectedDf] - f[cellDf];
				//		ftemp[cellDf] = f[advectedDf];
				//	}
				//	else if (cell_type[advectedCellIndex] == cell_types::interfase)
				//	{
				//		float currentEpsilon = getEpsilon(cellIndex), neighborEpsilon = getEpsilon(advectedCellIndex);
				//		deltaMass += (f[inverseAdvectedDf] - f[cellDf]) * (currentEpsilon + neighborEpsilon) / 2.0f;
				//		ftemp[cellDf] = f[advectedDf];
				//	}
				//	else if (cell_type[advectedCellIndex] == cell_types::gas)
				//	{
				//		ftemp[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
				//			feq[inverseSpeedDirectionIndex[l]] + feq[l] - f[l];
				//	}

				//	if (dot(cell_normal, speedDirection[l]) < 0.0f)
				//		ftemp[l] = feq[inverseSpeedDirectionIndex[l]] + feq[l] - f[l];
				//}
				
			}				

			//if (cell_type[advectedCellIndex] == cell_types::interfase)
			//	cell_mass_temp[cellIndex] = cell_mass[cellIndex] + deltaMass;
		}
		else
		{
			// Cycle to bounce the dfs when an obstacle is encountered
			for (int l = 0; l < 19; l++)
			{
				cellDf = cellIndex * _stride + l;
				advectedDf = cellIndex * _stride + inverseSpeedDirectionIndex[l];

				ftemp[cellDf] = f[advectedDf];
			}
		}
	}
}

// Collision remains the same. Just added a step to determine if a given cell filled or emptied.
void latticed3q19::collide(void)
{
	int cell_df = 0;
	float *feq;
	feq = new float[_stride]();
	//for (int i0 = 0; i0 < _numberLatticeElements; i0++)
	for (int k = 0; k<_depth; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i<_width; i++)
	{
		int cellIndex = I3D(_width, _height, i, j, k);
		if (cell_type[cellIndex] != cell_types::solid && cell_type[cellIndex] != cell_types::gas)
		{
			deriveQuantities(cellIndex);
			calculateEquilibriumFunction(feq, velocityVector[cellIndex], ro[cellIndex]);;

			for (int l = 0; l < 19; l++)
			{
				cell_df = cellIndex * _stride + l;
				f[cell_df] = (1.0f - _w) * ftemp[cell_df] + _w * feq[l]
				//	//To include gravity
				+ latticeWeights[l] * ro[cellIndex] * dot(speedDirection[l], float3{ 0, -latticeAcceleration, 0 });
			}				
			
			if (cell_type[cellIndex] == cell_types::interfase)
				setFilledEmptied(i, j, k);
		}
	}
}

void latticed3q19::calculateEquilibriumFunction(float *feq, float3 inVector, float inRo)
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

		feq[i] = w * (inRo + (3.0f * eiU) / (_vMax*_vMax) 
			- (1.5f * uSq) / (_vMax*_vMax) + (4.5f * eiUsq) / (_vMax*_vMax*_vMax*_vMax));
		//feq[i] = w * inRo * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

// Have to find how to correctly initialize the interfase cells. 
void latticed3q19::initLatticeDistributions()
{
	int cellDf = 0;
	float scaleInterfaceDf = 1.0f;

	for (int cellIndex = 0; cellIndex < _numberLatticeElements; cellIndex++)
	if (cell_type[cellIndex] != cell_types::solid && cell_type[cellIndex] != cell_types::gas)
	{
		for (int l = 0; l<_stride; l++)
		{
			cellDf = cellIndex*_stride + l;
			scaleInterfaceDf = cell_type[cellIndex] == cell_types::interfase ? 1.f : 1.0f;
			ftemp[cellDf] = f[cellDf] = latticeWeights[l] * scaleInterfaceDf;
		}

		deriveQuantities(cellIndex);

		if (cell_type[cellIndex] == cell_types::fluid)
			cell_mass[cellIndex] = cell_mass_temp[cellIndex] = ro[cellIndex];
		if (cell_type[cellIndex] == cell_types::interfase)
			cell_mass[cellIndex] = cell_mass_temp[cellIndex] = ro[cellIndex];
	}
}

// Calculates the density and speed of a cell
void latticed3q19::deriveQuantities(int index)
{
	int dfIndex = 0;

	// Calculate density and velocity
	ro[index] = 0;
	velocityVector[index] = float3{ 0, 0, 0 };

	for (int l = 0; l < _stride; l++)
	{
		ro[index] += f[index * _stride + l];
		dfIndex = index * _stride + l;
		velocityVector[index] += float3_ScalarMultiply(f[dfIndex], speedDirection[l]);
	}

	// On the original paper this operation is not done, however,  it is done on many other ones. 
	// This may be a point to chek in the future. 
	velocityVector[index] = float3_ScalarMultiply(1.0f / ro[index], velocityVector[index]);

	//float n = float3_Norm(velocityVector[index]);
	//if (n > _vMax)
	//	velocityVector[index] = float3_ScalarMultiply(_vMax / n, velocityVector[index]);
}

float latticed3q19::getEpsilon(int cellIndex)
{
	if (ro[cellIndex] > 0.0f)
		return cell_mass[cellIndex] / ro[cellIndex];
	else
		return 0.0f;
}

float3 latticed3q19::calculateNormal(int i, int j, int k)
{
	return float3
	{
	(getEpsilon(I3D(_width, _height, i - 1, j, k)) - getEpsilon(I3D(_width, _height, i + 1, j, k))) / 2,
	(getEpsilon(I3D(_width, _height, i , j - 1, k)) - getEpsilon(I3D(_width, _height, i , j + 1, k))) / 2,
	(getEpsilon(I3D(_width, _height, i , j, k - 1)) - getEpsilon(I3D(_width, _height, i , j, k + 1))) / 2
	};
}

// Have to check if cell_mass or cell_mass_temp is the correct value to check against. The same for ro.
void latticed3q19::setFilledEmptied(int i, int j, int k)
{
	float _k = 0.001f;

	int index = I3D(_width, _height, i, j, k);
	if (cell_mass_temp[index] > (1.0f + _k) * ro[index])
	{
		vector<int3>::iterator fillIterator =
			std::find(_filledCells.begin(), _filledCells.end(), int3{ i, j, k });
		if (fillIterator == _filledCells.end())
			_filledCells.push_back(int3{ i, j, k });
	}
	else if (cell_mass_temp[index] < (-_k) * ro[index])
	{
		vector<int3>::iterator emptyIterator =
			std::find(_emptiedCells.begin(), _emptiedCells.end(), int3{ i, j, k });
		if (emptyIterator == _emptiedCells.end())
			_emptiedCells.push_back(int3{ i, j, k });
	}
}

void latticed3q19::flagReinitialization()
{
	int nb_x, nb_y, nb_z;
	int cellIndex, nbCellIndex;

	for (int3 filledCell : _filledCells)
	{
		cellIndex = I3D(_width, _height, filledCell.x, filledCell.y, filledCell.z);

		for (int l = 0; l < _stride; l++)
		{
			nb_x = (filledCell.x + speedDirection[l].x);
			nb_y = (filledCell.y + speedDirection[l].y);
			nb_z = (filledCell.z + speedDirection[l].z);

			nbCellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);
			
			if (cell_type[nbCellIndex] == cell_types::gas)
			{
				cell_type_temp[nbCellIndex] = cell_types::interfase;
				averageSurroundings(nb_x, nb_y, nb_z);
			}
			else if (cell_type[nbCellIndex] == cell_types::interfase)
			{
				vector<int3>::iterator emptyIterator =
					std::find(_emptiedCells.begin(), _emptiedCells.end(), int3{ nb_x, nb_y, nb_z });
				if (emptyIterator != _emptiedCells.end())
					_emptiedCells.erase(emptyIterator);
			}
		}

		cell_type_temp[cellIndex] = cell_types::fluid;
	}

	for (int3 emptyCell : _emptiedCells)
	{
		cellIndex = I3D(_width, _height, emptyCell.x, emptyCell.y, emptyCell.z);

		for (int l = 0; l < _stride; l++)
		{
			nb_x = (emptyCell.x + speedDirection[l].x);
			nb_y = (emptyCell.y + speedDirection[l].y);
			nb_z = (emptyCell.z + speedDirection[l].z);

			nbCellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cell_type[nbCellIndex] == cell_types::fluid)
				cell_type_temp[nbCellIndex] = cell_types::interfase;
		}

		cell_type_temp[cellIndex] = cell_types::gas;
	}
	
	for (int l = 0; l < _numberLatticeElements; l++)
		cell_type[l] = cell_type_temp[l];
}

void latticed3q19::averageSurroundings(int i, int j, int k)
{
	int newI, newJ, newK;
	int cellIndex, nbCellIndex, numberNb = 0;

	float ro_avg = 0.0f;
	float3 velocity_avg = { 0.0f, 0.0f, 0.0f };
	float *feq = new float[_stride]();

	cellIndex = I3D(_width, _height, i, j, k);

	for (int l = 1; l < _stride; l++)
	{
		newI = (i + speedDirection[l].x);
		newJ = (j + speedDirection[l].y);
		newK = (k + speedDirection[l].z);

		nbCellIndex = I3D(_width, _height, newI, newJ, newK);

		if (cell_type[nbCellIndex] == cell_types::fluid || cell_type[nbCellIndex] == cell_types::interfase)
		{
			ro_avg += ro[nbCellIndex];
			velocity_avg += velocityVector[nbCellIndex];
			numberNb++;
		}
	}

	if (numberNb > 0)
	{
		ro[cellIndex] = ro_avg / numberNb;
		velocityVector[cellIndex] /= (float)numberNb;
	}

	calculateEquilibriumFunction(feq, velocityVector[cellIndex], ro[cellIndex]);

	for (int l = 0; l < _stride; l++)
		f[cellIndex*_stride + l] = feq[l];
}

void latticed3q19::distributeExcessMass()
{
	int nb_x, nb_y, nb_z;
	int cellIndex, nbCellIndex;

	float3 normal = { 0.0f, 0.0f, 0.0f };
	float excessMass = 0.0f, *vi = new float[_stride](), vTotal=0.0f;

	for (int3 filledCell : _filledCells)
	{
		cellIndex = I3D(_width, _height, filledCell.x, filledCell.y, filledCell.z);

		normal = calculateNormal(filledCell.x, filledCell.y, filledCell.z);

		excessMass = cell_mass[cellIndex] - ro[cellIndex];

		for (int l = 0; l < _stride; l++)
		{
			vi[l] = dot(speedDirection[l], normal);
			if (vi[l] <= 0.0f) vi[l] = 0.0f;
			vTotal += vi[l];
		}

		if (vTotal != 0.0f)
		for (int l = 0; l < _stride; l++)
		{
			nb_x = (filledCell.x + speedDirection[l].x);
			nb_y = (filledCell.y + speedDirection[l].y);
			nb_z = (filledCell.z + speedDirection[l].z);

			nbCellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cell_type[nbCellIndex] == cell_types::interfase)
				cell_mass_temp[nbCellIndex] = cell_mass[nbCellIndex] + excessMass * vi[l] / vTotal;
		}
	}

	excessMass = 0.0f, vi = new float[_stride](), vTotal = 0.0f;
	for (int3 emptiedCell : _emptiedCells)
	{
		cellIndex = I3D(_width, _height, emptiedCell.x, emptiedCell.y, emptiedCell.z);

		normal = calculateNormal(emptiedCell.x, emptiedCell.y, emptiedCell.z);

		float3 inverseNormal = float3_ScalarMultiply(-1.0f, normal);

		excessMass = cell_mass[cellIndex];

		for (int l = 0; l < _stride; l++)
		{
			vi[l] = dot(speedDirection[l], inverseNormal);
			if (vi[l] >= 0.0f) vi[l] = 0.0f;
			vTotal += vi[l];
		}

		if (vTotal != 0.0f)
		for (int l = 0; l < _stride; l++)
		{
			nb_x = (emptiedCell.x + speedDirection[l].x);
			nb_y = (emptiedCell.y + speedDirection[l].y);
			nb_z = (emptiedCell.z + speedDirection[l].z);

			nbCellIndex = I3D(_width, _height, nb_x, nb_y, nb_z);

			if (cell_type[nbCellIndex] == cell_types::interfase)
				cell_mass_temp[nbCellIndex] = cell_mass[nbCellIndex] + excessMass * vi[l] / vTotal;
		}
	}

	memcpy(cell_mass, cell_mass_temp, sizeof(float)*_numberLatticeElements);

	_filledCells.clear();
	_emptiedCells.clear();
}

void latticed3q19::reconstructInterfaceDfs()
{
	int cellIndex = 0, nbCellIndex = 0, newI = 0, newJ = 0, newK = 0;
	float *feq_air = new float[_stride]();

	for (int k = 0; k<_depth ; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i < _width; i++)
	{
		cellIndex = I3D(_width, _height, i, j, k);

		if (cell_type[cellIndex] ==cell_types::interfase)
		{
			for (int l = 0; l < 19; l++)
			{
				newI = (int)(i + speedDirection[l].x);
				newJ = (int)(j + speedDirection[l].y);
				newK = (int)(k + speedDirection[l].z);

				nbCellIndex = I3D(_width, _height, newI, newJ, newK);

				// Adjust the df that comes from a gas cell
				if (cell_type[nbCellIndex] == cell_types::gas)
				{
					calculateEquilibriumFunction(feq_air, velocityVector[cellIndex], 1.0f);
					ftemp[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
						feq_air[l] + feq_air[inverseSpeedDirectionIndex[l]] - f[cellIndex * _stride + l];
				}

				float3 normal = calculateNormal(i, j, k);
				
				// 
				if (dot(speedDirection[l], normal) < 0)
				{
					f[cellIndex * _stride + inverseSpeedDirectionIndex[l]] =
						feq_air[l] + feq_air[inverseSpeedDirectionIndex[l]] - f[cellIndex * _stride + l];
				}

			}
		}
	}
}

void latticed3q19::calculateMassExchange()
{
	int cellIndex = 0, nbCellIndex = 0, newI = 0, newJ = 0, newK = 0;
	float *delta_mass = new float[_stride](), delta_total = 0.0f;

	for (int k = 0; k <_depth -1; k++)
	for (int j = 0; j < _height -1; j++)
	for (int i = 0; i < _width -1; i++)
	{
		cellIndex = I3D(_width, _height, i, j, k);
		fill(delta_mass, delta_mass + _stride, 0.0f);

		if (cell_type[cellIndex] == cell_types::interfase)
		{
			for (int l = 0; l < 19; l++)
			{
				newI = (i + speedDirection[l].x);
				newJ = (j + speedDirection[l].y);
				newK = (k + speedDirection[l].z);

				nbCellIndex = I3D(_width, _height, newI, newJ, newK);

				if (cell_type[nbCellIndex] == cell_types::fluid)
					delta_mass[l] = f[nbCellIndex * _stride + inverseSpeedDirectionIndex[l]] - f[cellIndex * _stride + l];
				else if (cell_type[nbCellIndex] == cell_types::interfase)
				{
					float current_epsilon = getEpsilon(cellIndex), nb_epsilon = getEpsilon(nbCellIndex);
					delta_mass[l] = f[nbCellIndex * _stride + inverseSpeedDirectionIndex[l]] - f[cellIndex * _stride + l] *
						((nb_epsilon + current_epsilon) * 0.5f);
				}		

				delta_total += delta_mass[l];
			}
			cell_mass_temp[cellIndex] = cell_mass[cellIndex] + delta_total;
		}
	}

	memcpy(cell_mass, cell_mass_temp, sizeof(float)* _numberLatticeElements);
}

void latticed3q19::setNumberNeighbors()
{
	int cellIndex = 0, nbIndex = 0, newI = 0, newJ= 0, newK = 0;

	for (int k = 0; k<_depth; k++)
	for (int j = 0; j < _height; j++)
	for (int i = 0; i<_width; i++)
	{
		cellIndex = I3D(_width, _height, i, j, k);

		nFluidNbs[cellIndex] = 0;
		nGasNbs[cellIndex] = 0;
		nInterNbs[cellIndex] = 0;
		nSolidNbs[cellIndex] = 0;

		// Avoid the element itself
		for (int l = 1; l < _stride; l++)
		{
			newI = (int)(i + speedDirection[l].x);
			newJ = (int)(j + speedDirection[l].y);
			newK = (int)(k + speedDirection[l].z);

			if (newI >= 0 && newJ >= 0 && newK >= 0)
			{
				nbIndex = I3D(_width, _height, newI, newJ, newK);

				if (nbIndex >= 0)
				{
					if (cell_type[nbIndex] == cell_types::fluid)
						nFluidNbs[cellIndex]++;
					else if (cell_type[nbIndex] == cell_types::gas)
						nGasNbs[cellIndex]++;
					else if (cell_type[nbIndex] == cell_types::interfase)
						nInterNbs[cellIndex]++;
					else if (cell_type[nbIndex] == cell_types::solid)
						nSolidNbs[cellIndex]++;
				}
			}
		}
		if (nFluidNbs[cellIndex] + nGasNbs[cellIndex] + nInterNbs[cellIndex] + nSolidNbs[cellIndex] != 18)
			int test = 0;
	}
}