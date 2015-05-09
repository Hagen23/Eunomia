#include "Lattice_thrust.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_stride = 19;
	_numberLatticeElements = _width * _height * _depth;
	_numberAllElements = _stride * _numberLatticeElements;

	_c = (float)(1.0 / sqrt(3.0));

	initThrust();
}
	
latticed3q19::~latticed3q19()
{

}

void latticed3q19::step(void)
{
	latticeSolidIndexes_d = latticeSolidIndexes_h;
	velocityVector_d = velocityVector_h;

	stream();
	collide();

	velocityVector_h = velocityVector_d;
}

void latticed3q19::initThrust()
{	
	for (unsigned int k = 0; k < _depth; k++)
	{
		for (unsigned int j = 0; j < _height; j++)
		{
			for (unsigned int i = 0; i < _width; i++)
			{
				for (unsigned int l = 0; l < 19; l++)
					latticeIndexes_h.push_back(make_uint4(i, j, k, l));
			}
		}
	}

	latticeIndexes_d = latticeIndexes_h;
	
	f_d = thrust::device_vector<float>(_numberAllElements, 0);

	ftemp_d = thrust::device_vector<float>(_numberAllElements, 0);

	latticeSolidIndexes_h = thrust::host_vector<unsigned int>(_numberLatticeElements, 0);

	latticeSolidIndexes_d = thrust::device_vector<unsigned int>(_numberLatticeElements, 0);

	velocityVector_h = thrust::host_vector<float3>(_numberLatticeElements, make_float3(0,0,0));

	velocityVector_d = thrust::device_vector<float3>(_numberLatticeElements, make_float3(0,0,0));
}

void latticed3q19::stream()
{
	thrust::for_each(	
						latticeIndexes_d.begin(), latticeIndexes_d.end(),
						latticeStream(	
										thrust::raw_pointer_cast(f_d.data()),
										thrust::raw_pointer_cast(ftemp_d.data()),
										_width, _height, _depth, _stride,
										thrust::raw_pointer_cast(latticeSolidIndexes_d.data())
									)
					);

	thrust::copy(ftemp_d.begin(), ftemp_d.end(), f_d.begin());
}

void latticed3q19::collide(void)
{
	thrust::for_each(
						latticeIndexes_d.begin(), latticeIndexes_d.end(),
						latticeCollide(
										thrust::raw_pointer_cast(f_d.data()),
										thrust::raw_pointer_cast(velocityVector_d.data()),
										_width, _height, _stride, _tau,
										thrust::raw_pointer_cast(latticeSolidIndexes_d.data())
										)
					);
}

void latticed3q19::calculateInEquilibriumFunction(float3 _inVector, float inRo)
{
	thrust::for_each(
						latticeIndexes_d.begin(), latticeIndexes_d.end(),
						latticeInEq(
										thrust::raw_pointer_cast(f_d.data()),
										thrust::raw_pointer_cast(ftemp_d.data()),
										_inVector, _width, _height, _stride, inRo, _c
						)
					);
}