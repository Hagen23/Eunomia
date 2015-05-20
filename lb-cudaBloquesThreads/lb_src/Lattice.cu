#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd3q19[_numberElements];
	c = 1.0 / sqrt(3.0);
   iteracion = 0;
}
	
latticed3q19::~latticed3q19()
{
	delete[] latticeElements;
}

__constant__ int dims[3], xx[19], yy[19], zz[19];

///__global__ void stream(bool *sol, float *f, float *ftemp, int *xx, int *yy, int *zz)
__global__ void stream(bool *sol, float *f, float *ftemp)
{
   int i = blockIdx.x;
   int j = blockIdx.y;
   int k = blockIdx.z;
   int l = threadIdx.x;
   int i0, i1;
   int newI, newJ, newK;

   i0 = dims[0]*(j+dims[1]*k)+i;
   if(!sol[i0])
   {
      newI = ( i + xx[l] );
      newJ = ( j + yy[l] );
      newK = ( k + zz[l] );

      //Checking for exit boundaries
      if(newI > (dims[0] - 1)) newI = 0;
      else if(newI <= 0) newI = dims[0] - 1;
      if(newJ > (dims[1] - 1)) newJ = 0;
      else if(newJ <= 0) newJ = dims[1] - 1;
      if(newK > (dims[2] - 1)) newK = 0;
      else if(newK <= 0) newK = dims[2] - 1;

      i1 = dims[0]*(newJ+dims[1]*newK)+newI;
      ftemp[i0*19+l] = f[i1*19+l];
    }
}

void latticed3q19::step(void)
{
   int dimens[3] = {_width, _height, _depth};
   dim3 bloques(_width, _height, _depth);
   int sDx[19], sDy[19], sDz[19];
///   int *xx, *yy, *zz;
   bool solid[_width*_height*_depth], *sol;
   float ff[_width*_height*_depth*19], fftemp[_width*_height*_depth*19], *f, *ftemp;

   for (int p=0; p<19; p++)
   {
      sDx[p] = speedDirection[p].x;
      sDy[p] = speedDirection[p].y;
      sDz[p] = speedDirection[p].z;
   }
   for (int p=0; p<_width*_height*_depth; p++)
   {
      solid[p] = latticeElements[p].isSolid;
      for (int q=0; q<19; q++)
      {
        ff[p*19+q] = latticeElements[p].f[q];
        fftemp[p*19+q] = latticeElements[p].ftemp[q];
      }
   }
   cudaMalloc( (void**)&sol, _width*_height*_depth*sizeof(bool));
   cudaMalloc( (void**)&f, _width*_height*_depth*19*sizeof(float));
   cudaMalloc( (void**)&ftemp, _width*_height*_depth*19*sizeof(float));
   cudaMalloc( (void**)&xx, 19*sizeof(int));
   cudaMalloc( (void**)&yy, 19*sizeof(int));
   cudaMalloc( (void**)&zz, 19*sizeof(int));
   cudaMemcpy(sol, solid, _width*_height*_depth*sizeof(bool), cudaMemcpyHostToDevice);
   cudaMemcpy(f, ff, _width*_height*_depth*19*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(ftemp, fftemp, _width*_height*_depth*19*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(xx, sDx, 19*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(yy, sDy, 19*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(zz, sDz, 19*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol( dims, dimens, sizeof(int)*3 );
   cudaMemcpyToSymbol( xx, sDx, sizeof(int)*19 );
   cudaMemcpyToSymbol( yy, sDy, sizeof(int)*19 );
   cudaMemcpyToSymbol( zz, sDz, sizeof(int)*19 );

   // Llamada a kernel
   ///stream<<< bloques, 19 >>>(sol, f, ftemp, xx, yy, zz);
   stream<<< bloques, 19 >>>(sol, f, ftemp);

   cudaMemcpy(ff, ftemp, _width*_height*_depth*19*sizeof(float), cudaMemcpyDeviceToHost);
   for (int p=0; p<_width*_height*_depth; p++)
   {
      for (int q=0; q<19; q++)
      {
        latticeElements[p].f[q] = ff[p*19+q];
        latticeElements[p].ftemp[q] = fftemp[p*19+q];
      }
   }

	collide();
	applyBoundaryConditions();
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
//	float temp;
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
