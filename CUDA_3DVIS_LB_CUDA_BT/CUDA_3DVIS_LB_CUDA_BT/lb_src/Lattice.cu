# ifndef LATTICE_CU
# define LATTICE_CU

#include "Lattice.h"
#include <stdio.h>

__constant__ int 			dims[4], speedDirection_c[19*3];
__constant__ float		latticeWeights_c[19];


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

	initCUDA();

	c = 1.0 / sqrt(3.0);
}

void latticed3q19::initCUDA()
{
	int 		dimens[4] = {_width, _height, _depth, _stride};
	float		sD[19*3], lW[19];

	for(int i = 0; i < _stride; i++)
	{
		lW[i] = latticeWeights[i];
		sD[i*3] = speedDirection[i].x;
    sD[i*3+1] = speedDirection[i].y;
    sD[i*3+2] = speedDirection[i].z;
	}

	cudaMalloc((void**)&f_d, _numberAllElements*sizeof(float));
	cudaMalloc((void**)&ftemp_d, _numberAllElements*sizeof(float));
	cudaMalloc((void**)&solid_d, _numberLatticeElements*sizeof(float));
	cudaMalloc((void**)&velocityVector_d, _numberLatticeElements*sizeof(float3));

	cudaMemcpyToSymbol( dims, dimens, sizeof(int)*4);
	cudaMemcpyToSymbol( speedDirection_c, sD, sizeof(float)*19*3 );
	cudaMemcpyToSymbol( latticeWeights_c, lW, sizeof(float)*19 );
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
	stream();
	collide();
	//applyBoundaryConditions();
}

__global__ void stream_device(float *f, float* ftemp, unsigned int *solid)
{
	int iSolid, iBase, iAdvected;
	int newI, newJ, newK;

	int i = blockIdx.x, j = blockIdx.y, k = threadIdx.x;
	int l = threadIdx.y;

	iSolid = //blockDim.x * (j + blockDim.y * k) + i; 
					 I3D(dims[0], dims[1], i, j, k);

	if(solid[iSolid] == 0)
	{
		newI = (int)( i + speedDirection_c[l*3] );
		newJ = (int)( j + speedDirection_c[l*3+1] );
		newK = (int)( k + speedDirection_c[l*3+2] );

		//Checking for exit boundaries
		if (newI >(dims[0]- 1)) newI = 0;
		else if (newI <= 0) newI = dims[0] - 1;

		if (newJ > (dims[1] - 1)) newJ = 0;
		else if (newJ <= 0) newJ = dims[1] - 1;

		if (newK > (dims[2] - 1)) newK = 0;
		else if (newK <= 0) newK = dims[2] - 1;

		iBase = iSolid*dims[3] + l;
		iAdvected = //(blockDim.x * (newJ + blockDim.y * newK) + newI) * gridDim.y + l;
								I3D_S(dims[0], dims[1], dims[3],  newI, newJ , newK, l );

		ftemp[iBase] = f[iAdvected];
	}
}

void latticed3q19::stream()
{
	cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	dim3 	blocks(_width, _height, 1);
	dim3	threads(_depth, 19,1);

	cudaMemcpy(f_d, f, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ftemp_d, ftemp, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(solid_d, solid, _numberLatticeElements*sizeof(unsigned int), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

	stream_device<<<blocks, threads>>>(f_d, ftemp_d, solid_d);

	cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("Time for stream: %f ms\n", time);
	
	cudaMemcpy(f, ftemp_d, _numberAllElements*sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ void calculateSpeedVector_device(int index, float *f, float &ro, float3 *velocityVector_d)
{
	float rovx = 0, rovy = 0, rovz = 0; 
	int i0 = 0;

	for (int i = 0; i<dims[3]; i++)
	{
		i0 = index * dims[3] + i;
		ro += f[i0];
		rovx += f[i0] * speedDirection_c[i*3];
		rovy += f[i0] * speedDirection_c[i*3+1];
		rovz += f[i0] * speedDirection_c[i*3+2];
	}

	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
	if (ro == ro && ro != 0.0)
	{
		velocityVector_d[index].x = rovx / ro;
		velocityVector_d[index].y = rovy / ro;
		velocityVector_d[index].z = rovz / ro;
	}
	else
	{
		velocityVector_d[index].x = 0;
		velocityVector_d[index].y = 0;
		velocityVector_d[index].z = 0;
	}
}

__device__ void calculateEquilibriumFunction_device(int index, float *feq, int ro, float c, float3 *velocityVector_d)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(velocityVector_d[index], velocityVector_d[index]);	//Velocity squared

	for (int i = 0; i<dims[3]; i++)
	{
		w = latticeWeights_c[i];
		eiU = dot(speedDirection_c[i*3], speedDirection_c[i*3+1], speedDirection_c[i*3+2], velocityVector_d[index]);
		eiUsq = eiU * eiU;

		feq[i] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

__device__ void solid_BC_device(int i0, float *f)
{
	float temp;

	temp = f[i0*dims[3]+1]; 	f[i0*dims[3]+1] = f[i0*dims[3]+2];		f[i0*dims[3]+2] = temp;		// f1	<-> f2
	temp = f[i0*dims[3]+3];		f[i0*dims[3]+3] = f[i0*dims[3]+4];		f[i0*dims[3]+4] = temp;		// f3	<-> f4
	temp = f[i0*dims[3]+5];		f[i0*dims[3]+5] = f[i0*dims[3]+6];		f[i0*dims[3]+6] = temp;		// f5	<-> f6
	temp = f[i0*dims[3]+7];		f[i0*dims[3]+7] = f[i0*dims[3]+12];		f[i0*dims[3]+12] = temp;		// f7	<-> f12
	temp = f[i0*dims[3]+8];		f[i0*dims[3]+8] = f[i0*dims[3]+11];		f[i0*dims[3]+11] = temp;		// f8	<-> f11
	temp = f[i0*dims[3]+9];		f[i0*dims[3]+9] = f[i0*dims[3]+14];		f[i0*dims[3]+14] = temp;		// f9	<-> f14
	temp = f[i0*dims[3]+10];	f[i0*dims[3]+10] = f[i0*dims[3]+13];	f[i0*dims[3]+13] = temp;		// f10	<-> f13
	temp = f[i0*dims[3]+15];	f[i0*dims[3]+15] = f[i0*dims[3]+18];	f[i0*dims[3]+18] = temp;		// f15	<-> f18
	temp = f[i0*dims[3]+16];	f[i0*dims[3]+16] = f[i0*dims[3]+17];	f[i0*dims[3]+17] = temp;		// f16	<-> f17
}

__global__ void collide_device(float *f, unsigned int *solid, float tau, float c, float3 *velocityVector_d)
{
	int 	index = I3D(dims[0], dims[1], blockIdx.x, blockIdx.y, threadIdx.x);
	int		iBase;
	float feq[19], ro;

	if(solid[index] == 0)
	{
		calculateSpeedVector_device(index, f, ro, velocityVector_d);
		calculateEquilibriumFunction_device(index, feq, ro, c, velocityVector_d);
		
//			for (int l = 0; l < 19; l++)
//			{
//				iBase = index * dims[3] + l;
//				f[iBase] = f[iBase] - (f[iBase] - feq[l]) / tau;
//			}
		iBase = index * dims[3] + threadIdx.y;
		f[iBase] = f[iBase] - (f[iBase] - feq[threadIdx.y]) / tau;
	}
	else
		solid_BC_device(index, f);
}

void latticed3q19::collide(void)
{
	cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	dim3 	blocks(_width, _height, 1);
	dim3	threads(_depth, 19,1);

	cudaMemcpy(f_d, f, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(solid_d, solid, _numberLatticeElements*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(velocityVector_d, velocityVector, _numberLatticeElements*sizeof(float3), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

	collide_device<<<blocks, threads>>>(f_d, solid_d, _tau, c, velocityVector_d);

	cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("Time for collide: %f ms\n", time);
	
	cudaMemcpy(f, ftemp_d, _numberAllElements*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(velocityVector, velocityVector_d, _numberLatticeElements*sizeof(float3), cudaMemcpyDeviceToHost);
}

void latticed3q19::applyBoundaryConditions()
{
	//in_BC(vector3d(0.0,0.0, -0.6));
}

void latticed3q19::solid_BC(int i0)
{
	float temp;

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

		ftemp[iBase] = f[iBase] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

#endif
