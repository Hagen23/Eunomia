#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd3q19[_numberElements];
	c = 1.0 / sqrt(3.0);
}
	
latticed3q19::~latticed3q19()
{
	delete[] latticeElements;
}

void latticed3q19::step(void)
{
	stream();
	collide();
	applyBoundaryConditions();
}

void latticed3q19::stream()
{
	int i0, i1;
	int newI, newJ, newK;

	for(int k =0; k<_depth; k++)
	{
		for(int j = 0; j < _height; j++)
		{
			for(int i = 0; i<_width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				if(!latticeElements[i0].isSolid)
					for(int l = 0; l < 19; l++)
					{
						newI = (int)( i + speedDirection[l].x );
						newJ = (int)( j + speedDirection[l].y );
						newK = (int)( k + speedDirection[l].z );
						
						//Checking for exit boundaries
						if(newI > (_width - 1)) newI = 0;
						else if(newI <= 0) newI = _width - 1;
						
						if(newJ > (_height - 1)) newJ = 0;
						else if(newJ <= 0) newJ = _height - 1;

						if(newK > (_depth - 1)) newK = 0;
						else if(newK <= 0) newK = _depth - 1;

						//if(newI> (_width-1) || newI<=0) 
						//	newI = (int)( i - speedDirection[l].x );
						//if(newJ> (_height-1) || newJ<=0) 
						//	newJ = (int)( j - speedDirection[l].y );
						//if(newK> (_depth-1) || newK<=0) 
						//	newK = (int)( k - speedDirection[l].z );

						i1 = I3D(_width, _height, newI, newJ , newK );

						latticeElements[i0].ftemp[l] = latticeElements[i1].f[l];
					}
			}
		}
	}

	for(int i = 0; i < _numberElements; i++)
		for(int j = 0; j < 19; j++)
			latticeElements[i].f[j] = latticeElements[i].ftemp[j];
}

void latticed3q19::collide(void)
{
	for (int i0 = 0; i0 < _numberElements; i0++)
	{
		if (!latticeElements[i0].isSolid)
		{
			latticeElements[i0].calculateSpeedVector();
			latticeElements[i0].calculateEquilibriumFunction();

			for (int l = 0; l < 19; l++)
				latticeElements[i0].f[l] = latticeElements[i0].f[l] - (latticeElements[i0].f[l] - latticeElements[i0].feq[l]) / _tau;
			//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
		}
		else
			solid_BC(i0);
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

void latticed3q19::top_bottom_boundary(void)
{
	float N_x, N_y, N_z;
	int i0;

	for (int k = 0; k < _depth; k + _depth - 1)
	{
		for (int j = 0; j < _height; j++)
		{
			for (int i = 0; i < _width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				N_x = 0.5 *
					(latticeElements[i0].f[1] + latticeElements[i0].f[7] + latticeElements[i0].f[8]
					- (latticeElements[i0].f[2] + latticeElements[i0].f[11] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.z;

				N_y = 0.5 *
					(latticeElements[i0].f[3] + latticeElements[i0].f[7] + latticeElements[i0].f[11]
					- (latticeElements[i0].f[4] + latticeElements[i0].f[8] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.y;

				if (k == 0)
				{
					latticeElements[i0].f[5] =
						latticeElements[i0].f[6] + 2.0 * latticeWeights[5] * latticeElements[i0].getRo() *
						latticeElements[i0].velocityVector.z / (c * c);

					latticeElements[i0].f[9] =
						latticeElements[i0].f[14] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
						latticeElements[i0].velocityVector.x) / 6.f - N_x;

					latticeElements[i0].f[13] =
						latticeElements[i0].f[10] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
						latticeElements[i0].velocityVector.x) / 6.f + N_x;

					latticeElements[i0].f[15] =
						latticeElements[i0].f[18] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
						latticeElements[i0].velocityVector.y) / 6.f - N_y;

					latticeElements[i0].f[17] =
						latticeElements[i0].f[16] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
						latticeElements[i0].velocityVector.y) / 6.f - N_y;
				}
				else
					{
						latticeElements[i0].f[6] =
							latticeElements[i0].f[5] + 2.0 * latticeWeights[5] * latticeElements[i0].getRo() *
							latticeElements[i0].velocityVector.z / (c * c);

						latticeElements[i0].f[9] =
							latticeElements[i0].f[14] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
							latticeElements[i0].velocityVector.x) / 6.f - N_x;

						latticeElements[i0].f[13] =
							latticeElements[i0].f[10] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
							latticeElements[i0].velocityVector.x) / 6.f + N_x;

						latticeElements[i0].f[15] =
							latticeElements[i0].f[18] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
							latticeElements[i0].velocityVector.y) / 6.f - N_y;

						latticeElements[i0].f[17] =
							latticeElements[i0].f[16] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
							latticeElements[i0].velocityVector.y) / 6.f - N_y;
					}
			}
		}
	}
}

void latticed3q19::left_right_boundary(void)
{
	float N_x, N_y, N_z;
	int i0;

	for (int k = 0; k < _depth; k + _depth - 1)
	{
		for (int j = 0; j < _height; j++)
		{
			for (int i = 0; i < _width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				N_x = 0.5 *
					(latticeElements[i0].f[1] + latticeElements[i0].f[7] + latticeElements[i0].f[8]
					- (latticeElements[i0].f[2] + latticeElements[i0].f[11] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.z;

				N_y = 0.5 *
					(latticeElements[i0].f[3] + latticeElements[i0].f[7] + latticeElements[i0].f[11]
					- (latticeElements[i0].f[4] + latticeElements[i0].f[8] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.y;

				latticeElements[i0].f[5] =
					latticeElements[i0].f[6] + 2.0 * latticeWeights[5] * latticeElements[i0].getRo() *
					latticeElements[i0].velocityVector.z / (c * c);

				latticeElements[i0].f[9] =
					latticeElements[i0].f[14] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
					latticeElements[i0].velocityVector.x) / 6.f - N_x;

				latticeElements[i0].f[13] =
					latticeElements[i0].f[10] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
					latticeElements[i0].velocityVector.x) / 6.f + N_x;

				latticeElements[i0].f[15] =
					latticeElements[i0].f[18] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
					latticeElements[i0].velocityVector.y) / 6.f - N_y;

				latticeElements[i0].f[17] =
					latticeElements[i0].f[16] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
					latticeElements[i0].velocityVector.y) / 6.f - N_y;
			}
		}
	}
}

void latticed3q19::front_back_boundary(void)
{
	float N_x, N_y, N_z;
	int i0;

	for (int k = 0; k < _depth; k + _depth - 1)
	{
		for (int j = 0; j < _height; j++)
		{
			for (int i = 0; i < _width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				N_x = 0.5 *
					(latticeElements[i0].f[1] + latticeElements[i0].f[7] + latticeElements[i0].f[8]
					- (latticeElements[i0].f[2] + latticeElements[i0].f[11] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.z;

				N_y = 0.5 *
					(latticeElements[i0].f[3] + latticeElements[i0].f[7] + latticeElements[i0].f[11]
					- (latticeElements[i0].f[4] + latticeElements[i0].f[8] + latticeElements[i0].f[12]))
					- 1.f / 3.f * latticeElements[i0].getRo() * latticeElements[i0].velocityVector.y;

				latticeElements[i0].f[5] =
					latticeElements[i0].f[6] + 2.0 * latticeWeights[5] * latticeElements[i0].getRo() *
					latticeElements[i0].velocityVector.z / (c * c);

				latticeElements[i0].f[9] =
					latticeElements[i0].f[14] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
					latticeElements[i0].velocityVector.x) / 6.f - N_x;

				latticeElements[i0].f[13] =
					latticeElements[i0].f[10] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
					latticeElements[i0].velocityVector.x) / 6.f + N_x;

				latticeElements[i0].f[15] =
					latticeElements[i0].f[18] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z +
					latticeElements[i0].velocityVector.y) / 6.f - N_y;

				latticeElements[i0].f[17] =
					latticeElements[i0].f[16] + latticeElements[i0].getRo() * (latticeElements[i0].velocityVector.z -
					latticeElements[i0].velocityVector.y) / 6.f - N_y;
			}
		}
	}
}

void latticed3q19::boundary_BC(vector3d inVector)
{
	top_bottom_boundary();
	left_right_boundary();
	front_back_boundary();
}

void latticed3q19::solid_BC(int i0)
{
	double temp;

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