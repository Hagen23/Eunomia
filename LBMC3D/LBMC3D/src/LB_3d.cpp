#include "LB_3d.h"


LB_d3_q19::LB_d3_q19(int width, int height, int depth, float tau)
{
	w1 = 1.0f/18.0f;
	w2 = 1.0f/36.0f;
	w3 = 1.0f/3.0f;
	
	_latticeWidth = width;
	_latticeHeight = height;
	_latticeDepth = depth;
	_tau = tau;
	
	lattice = new latticed3q19(width, height, depth);
	plotvar = new float[lattice->getNumElements()];
}

LB_d3_q19::~LB_d3_q19(void)
{
	delete lattice;
	delete plotvar;
}

void LB_d3_q19::calculateSpeedVectors(void)
{
	for(int i =0; i<lattice->getNumElements(); i++)
		lattice->latticeElements[i].calculateSpeedVector();
}

void LB_d3_q19::equilibriumFunction(void)
{
	float w = 0,  ev = 0;
	
	for(int i =0; i<lattice->getNumElements(); i++)
	{
		for(int j = 0; j<19; j++)
		{
			if((j>=0 && j <=3) || j ==16 || j ==17) w = w1;
			if(j>=4 && j <=15) w = w2;
			if(j==18) w = w3;

			//ev = speedDirection[j][0] * vx + speedDirection[j][1] * vy  + speedDirection[j][2] * vz;
			//lattice->latticeElements[i].velocities[j] = w * ro * (1.f + 3.f*ev + 4.5f*ev*ev - v_sq_term );
		}
	}
}
