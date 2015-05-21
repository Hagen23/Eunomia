#pragma once

#ifndef LATTICE_CU
#define LATTICE_CU

#include "Lattice_thrust.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_stride = 19;
	_numberLatticeElements = _width * _height * _depth;
	_numberAllElements = _stride * _numberLatticeElements;
	_c = 1 / sqrtf(3.0f);

	initThrust();
}

latticed3q19::~latticed3q19()
{

}

void latticed3q19::step(void)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	latticeSolidIndexes_d = latticeSolidIndexes_h;
	velocityVector_d = velocityVector_h;

	cudaEventRecord(start, 0);

	stream();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\nTime for stream: %f ms\n", time);

	cudaEventRecord(start, 0);

	collide();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("Time for collide: %f ms\n", time);

	velocityVector_h = velocityVector_d;
}

void latticed3q19::initThrust()
{

	for (unsigned int i = 0; i < 19 * 3; i++)
	{
		speedDirection_d.push_back(speedDirection_in[i]);
	}
	speedDirection_d_ptr = thrust::raw_pointer_cast(speedDirection_d.data());

	for (unsigned int i = 0; i < 19; i++)
	{
		latticeWeights_d.push_back(latticeWeights_in[i]);
	}
	latticeWeights_d_ptr = thrust::raw_pointer_cast(latticeWeights_d.data());


	try
	{
		f_d = thrust::device_vector<float>(_numberAllElements, 0.0f);
	}
	catch (thrust::system_error& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}


	for (unsigned int k = 0; k < _depth; k++)
	{
		for (unsigned int j = 0; j < _height; j++)
		{
			for (unsigned int i = 0; i < _width; i++)
			{
				for (unsigned int w = 0; w < 19; w++)
				{
					//latticeIndexes_h.push_back(make_uint4(i, j, k, w));
					latticeIndexes_hi.push_back(i);
					latticeIndexes_hj.push_back(j);
					latticeIndexes_hk.push_back(k);
					latticeIndexes_hw.push_back(w);
				}
			}
		}
	}

	latticeIndexes_di = latticeIndexes_hi;
	latticeIndexes_dj = latticeIndexes_hj;
	latticeIndexes_dk = latticeIndexes_hk;
	latticeIndexes_dw = latticeIndexes_hw;

	ftemp_d = thrust::device_vector<float>(_numberAllElements, 0.0f);

	latticeSolidIndexes_h = thrust::host_vector<unsigned int>(_numberLatticeElements, 0);

	latticeSolidIndexes_d = thrust::device_vector<unsigned int>(_numberLatticeElements, 0);

	velocityVector_h = thrust::host_vector<float>(_numberLatticeElements * 3, 0.0f);

	velocityVector_d = thrust::device_vector<float>(_numberLatticeElements * 3, 0.0f);

	latticeWeights_d = latticeWeights_h;
}

void latticed3q19::stream()
{
	float* f_d_ptr = thrust::raw_pointer_cast(f_d.data());
	float* ftemp_d_ptr = thrust::raw_pointer_cast(ftemp_d.data());
	unsigned int* lsi_ptr = thrust::raw_pointer_cast(latticeSolidIndexes_d.data());
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.begin(),
				latticeIndexes_dj.begin(),
				latticeIndexes_dk.begin(),
				latticeIndexes_dw.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.end(),
				latticeIndexes_dj.end(),
				latticeIndexes_dk.end(),
				latticeIndexes_dw.end()
			)
		),
		latticeStream(
			f_d_ptr, ftemp_d_ptr, speedDirection_d_ptr,
			_width, _height, _depth, _stride,
			lsi_ptr
		)
	);
	thrust::copy(ftemp_d.begin(), ftemp_d.end(), f_d.begin());
}

void latticed3q19::collide(void)
{
	float* f_d_ptr = thrust::raw_pointer_cast(f_d.data());
	float* vv_ptr = thrust::raw_pointer_cast(velocityVector_d.data());
	unsigned int* lsi_ptr = thrust::raw_pointer_cast(latticeSolidIndexes_d.data());
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.begin(),
				latticeIndexes_dj.begin(),
				latticeIndexes_dk.begin(),
				latticeIndexes_dw.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.end(),
				latticeIndexes_dj.end(),
				latticeIndexes_dk.end(),
				latticeIndexes_dw.end()
			)
		),
		latticeCollide(
			f_d_ptr,
			vv_ptr,
			speedDirection_d_ptr,
			latticeWeights_d_ptr,
			_width, _height, _stride, _tau,
			lsi_ptr
		)
	);
}

void latticed3q19::calculateInEquilibriumFunction(float3 _inVector, float inRo)
{
	thrust::for_each(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.begin(),
				latticeIndexes_dj.begin(),
				latticeIndexes_dk.begin(),
				latticeIndexes_dw.begin()
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
				latticeIndexes_di.end(),
				latticeIndexes_dj.end(),
				latticeIndexes_dk.end(),
				latticeIndexes_dw.end()
			)
		),
		latticeInEq(
			thrust::raw_pointer_cast(f_d.data()),
			thrust::raw_pointer_cast(ftemp_d.data()),
			_inVector, _width, _height, _stride, inRo, _c
		)
	);
}
#endif
