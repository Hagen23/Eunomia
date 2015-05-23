#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_stride = 19;
	_numberLatticeElements = _width * _height * _depth;
	_numberAllElements = _stride * _numberLatticeElements;

	f = new float[_numberAllElements]();
	ftemp = new float[_numberAllElements]();
	feq = new float[_stride]();
	solid = new unsigned int[_numberLatticeElements]();
	velocityVector = new float3[_numberLatticeElements]();

	c = 1.0 / sqrt(3.0);
}
	
latticed3q19::~latticed3q19()
{
	delete[] f;
	delete[] ftemp;
	delete[] feq;
	delete[] solid;
	delete[] velocityVector;
}

void latticed3q19::step(void)
{
	float time;
	clock_t begin = clock();

	stream();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	
	printf("Stream: %f ms\n", elapsed_secs * 1000);
	begin = clock();

	collide();

	end = clock(); 
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Collide: %f ms\n", elapsed_secs * 1000);

	applyBoundaryConditions();
}

void latticed3q19::stream()
{
	int iSolid, iBase, iAdvected;
	int newI, newJ, newK;

	for(int k =0; k<_depth; k++)
	{
		for(int j = 0; j < _height; j++)
		{
			for(int i = 0; i<_width; i++)
			{
				iSolid = I3D(_width, _height, i, j, k);

				if(solid[iSolid] == 0)
					for(int l = 0; l < 19; l++)
					{
						newI = (int)( i + speedDirection[l].x );
						newJ = (int)( j + speedDirection[l].y );
						newK = (int)( k + speedDirection[l].z );

						//Checking for exit boundaries
						if (newI >(_width - 1)) newI = 0;
						else if (newI <= 0) newI = _width - 1;

						if (newJ > (_height - 1)) newJ = 0;
						else if (newJ <= 0) newJ = _height - 1;

						if (newK > (_depth - 1)) newK = 0;
						else if (newK <= 0) newK = _depth - 1;

						iBase = iSolid*_stride + l;
						iAdvected = I3D_S(_width, _height, _stride,  newI, newJ , newK, l );

						ftemp[iBase] = f[iAdvected];
					}
			}
		}
	}

	for (int i0 = 0; i0 < _numberAllElements; i0++)
		f[i0] = ftemp[i0];
}

void latticed3q19::collide(void)
{
	int iBase = 0; 
	for (int i0 = 0; i0 < _numberLatticeElements; i0++)
	{
		if (solid[i0] == 0)
		{
			calculateSpeedVector(i0);
			calculateEquilibriumFunction(i0);

			for (int l = 0; l < 19; l++)
			{
				iBase = i0 * _stride + l;
				f[iBase] = f[iBase] - (f[iBase] - feq[l]) / _tau;
			}
			
			//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
		}
		else
			solid_BC(i0);
	}
}

void latticed3q19::applyBoundaryConditions()
{
	//in_BC(vector3d(0.0,0.0, -0.6));
}

void latticed3q19::solid_BC(int i0)
{
	double temp;

	temp = f[i0*_stride+1]; 	f[i0*_stride+1] = f[i0*_stride+2];		f[i0*_stride+2] = temp;		// f1	<-> f2
	temp = f[i0*_stride+3];		f[i0*_stride+3] = f[i0*_stride+4];		f[i0*_stride+4] = temp;		// f3	<-> f4
	temp = f[i0*_stride+5];		f[i0*_stride+5] = f[i0*_stride+6];		f[i0*_stride+6] = temp;		// f5	<-> f6
	temp = f[i0*_stride+7];		f[i0*_stride+7] = f[i0*_stride+12];		f[i0*_stride+12] = temp;		// f7	<-> f12
	temp = f[i0*_stride+8];		f[i0*_stride+8] = f[i0*_stride+11];		f[i0*_stride+11] = temp;		// f8	<-> f11
	temp = f[i0*_stride+9];		f[i0*_stride+9] = f[i0*_stride+14];		f[i0*_stride+14] = temp;		// f9	<-> f14
	temp = f[i0*_stride+10];	f[i0*_stride+10] = f[i0*_stride+13];	f[i0*_stride+13] = temp;		// f10	<-> f13
	temp = f[i0*_stride+15];	f[i0*_stride+15] = f[i0*_stride+18];	f[i0*_stride+18] = temp;		// f15	<-> f18
	temp = f[i0*_stride+16];	f[i0*_stride+16] = f[i0*_stride+17];	f[i0*_stride+17] = temp;		// f16	<-> f17
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
void latticed3q19::calculateSpeedVector(int index)
{
	//calculateRo();
	//rovx = rovy = rovz = 0; 

	ro = rovx = rovy = rovz = 0;
	int i0 = 0;
	for (int i = 0; i<_stride; i++)
	{
		i0 = index * _stride + i;
		ro += f[i0];
		rovx += f[i0] * speedDirection[i].x;
		rovy += f[i0] * speedDirection[i].y;
		rovz += f[i0] * speedDirection[i].z;
	}

	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
	if (ro == ro && ro != 0.0)
	{
		velocityVector[index].x = rovx / ro;
		velocityVector[index].y = rovy / ro;
		velocityVector[index].z = rovz / ro;
	}
	else
	{
		velocityVector[index].x = 0;
		velocityVector[index].y = 0;
		velocityVector[index].z = 0;
	}
}

void latticed3q19::calculateEquilibriumFunction(int index)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(velocityVector[index], velocityVector[index]);	//Velocity squared

	for (int i = 0; i<_stride; i++)
	{
		w = latticeWeights[i];
		eiU = dot(speedDirection[i], velocityVector[index]);
		eiUsq = eiU * eiU;
		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		feq[i] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

void latticed3q19::calculateInEquilibriumFunction(int index, float3 inVector, float inRo)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(inVector, inVector);	//Velocity squared

	int iBase = 0; 
	ro = inRo;

	for (int i = 0; i<_stride; i++)
	{
		w = latticeWeights[i];
		eiU = dot(speedDirection[i], inVector);
		eiUsq = eiU * eiU;

		iBase = index*_stride + i;
		//ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		ftemp[iBase] = f[iBase] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}