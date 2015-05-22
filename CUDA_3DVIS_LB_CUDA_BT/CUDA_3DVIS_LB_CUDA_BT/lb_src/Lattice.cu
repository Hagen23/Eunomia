# ifndef LATTICE_CU
# define LATTICE_CU

#include "Lattice.h"
#include <stdio.h>

__constant__ int 		dims[4];
__constant__ float		latticeWeights_c[19], speedDirection_c[19 * 3];;

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
	velocityVector = new float[_numberLatticeElements*3]();

	outputFile.open("cuda_global_times_1.cvs");
	outputFile << "Lattice Width " << _width << ";Lattice Height " << _height << ";Lattice Depth " << _depth << ";Total cells" << std::endl;
	outputFile << _width << ";" << _height << ";" << _depth << ";" << _numberLatticeElements << std::endl << std::endl;
	outputFile << "Stream;Collide.\n";
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

	checkCudaErrors(cudaMalloc((void**)&f_d, _numberAllElements*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&ftemp_d, _numberAllElements*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&solid_d, _numberLatticeElements*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&velocityVector_d, _numberLatticeElements*sizeof(float) * 3));

	checkCudaErrors(cudaMemcpyToSymbol(dims, dimens, sizeof(int) * 4));
	checkCudaErrors(cudaMemcpyToSymbol(speedDirection_c, sD, sizeof(float) * 19 * 3));
	checkCudaErrors(cudaMemcpyToSymbol(latticeWeights_c, lW, sizeof(float) * 19));
}
	
latticed3q19::~latticed3q19()
{
	delete[] f;
	delete[] ftemp;
	delete[] feq;
	delete[] solid;
	delete[] velocityVector;

	outputFile.close();
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

	checkCudaErrors(cudaMemcpy(f_d, f, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ftemp_d, ftemp, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(solid_d, solid, _numberLatticeElements*sizeof(unsigned int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(start, 0));

	stream_device<<<blocks, threads>>>(f_d, ftemp_d, solid_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	//printf("Time for stream: %f ms\n", time);
	outputFile << time << ";";
	
	checkCudaErrors(cudaMemcpy(f, ftemp_d, _numberAllElements*sizeof(float), cudaMemcpyDeviceToHost));
}

__device__ float calculateSpeedVector_device(int index, float *f, float *velocityVector_d)
{
	float ro = 0, rovx = 0, rovy = 0, rovz = 0; 
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
		velocityVector_d[index*3] = rovx / ro;
		velocityVector_d[index*3+1] = rovy / ro;
		velocityVector_d[index*3+2] = rovz / ro;
	}
	else
	{
		printf("ERROR!\n");
		velocityVector_d[index*3] = 0;
		velocityVector_d[index*3+1] = 0;
		velocityVector_d[index*3+2] = 0;
	}

	return ro;
}

__device__ void calculateEquilibriumFunction_device(int index, float *feq, int ro, float c, float *velocityVector_d)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = dot(velocityVector_d[index * 3], velocityVector_d[index * 3 + 1], velocityVector_d[index * 3 +2],
					velocityVector_d[index * 3], velocityVector_d[index * 3 + 1], velocityVector_d[index * 3 + 2]);	//Velocity squared

	for (int i = 0; i<dims[3]; i++)
	{
		w = latticeWeights_c[i];
		eiU = dot(speedDirection_c[i*3], speedDirection_c[i*3+1], speedDirection_c[i*3+2], 
					velocityVector_d[index * 3], velocityVector_d[index * 3 + 1], velocityVector_d[index * 3 + 2]);
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

__global__ void collide_device(float *f, unsigned int *solid, float tau, float c, float *velocityVector_d)
{
	int 	index = I3D(dims[0], dims[1], blockIdx.x, blockIdx.y, threadIdx.x);
	int		iBase;
	float	feq[19], ro;

	if(solid[index] == 0)
	{
		ro = calculateSpeedVector_device(index, f, velocityVector_d);
		calculateEquilibriumFunction_device(index, feq, ro, c, velocityVector_d);

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

	checkCudaErrors(cudaMemcpy(f_d, f, _numberAllElements*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(velocityVector_d, velocityVector, _numberLatticeElements*sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(start, 0));

	collide_device<<<blocks, threads>>>(f_d, solid_d, _tau, c, velocityVector_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	//printf("Time for collide: %f ms\n", time);
	outputFile << time << "\n";
	
	checkCudaErrors(cudaMemcpy(f, f_d, _numberAllElements*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(velocityVector, velocityVector_d, _numberLatticeElements*sizeof(float3), cudaMemcpyDeviceToHost));
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
