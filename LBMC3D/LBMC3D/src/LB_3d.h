/*
D3Q19 lattice
*/

#ifndef LB_3D
#define LB_3D

#pragma once

#include "Lattice.h"

class LB_d3_q19
{

private:
	float _latticeWidth, _latticeHeight, _latticeDepth;
	
	//weights for the equilibrium function; w1 for i=[0,3] and [16,17]; w2 for i=[4,15], w3 for i=18
	float w1, w2, w3; 
	float _tau;
	float rtau, rtau1;

	void calculateSpeedVectors(void);
	void equilibriumFunction(void);

public:
	latticed3q19 *lattice;
	float *plotvar;

	LB_d3_q19(int width, int height, int depth, float tau);
	~LB_d3_q19(void);
};

#endif // !LB_3D