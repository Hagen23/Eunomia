/*
D3Q19 lattice
*/


#ifndef LB_3D
#define LB_3D

#pragma once
class LB_3d
{

private:
	int _vVectorSize;
	float latticeWidth, latticeHeight, latticeDepth;
	float *lattice;

public:
	LB_3d(int vVectorSize);
	~LB_3d(void);
};

#endif // !LB_3D