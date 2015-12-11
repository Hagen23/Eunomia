/**
*@class LB_Octree
*Creates a LB lattice based off an Octree. It uses the D3Q19 model.
*
*/

#pragma once

#ifndef LATTICE_BOLTZMANN_OCTREE
#define LATTICE_BOLTZMANN_OCTREE

#include <HTRBasicDataStructures.h>
#include <OctreeGenerator.h>
#include <vector>

namespace htr
{
	static htr::Point3D speedDirection[19] =
	{
		htr::Point3D(0, 0, 0),
		htr::Point3D(1, 0, 0), htr::Point3D(-1, 0, 0), htr::Point3D(0, 1, 0), htr::Point3D(0, -1, 0),
		htr::Point3D(0, 0, 1), htr::Point3D(0, 0, -1), htr::Point3D(1, 1, 0), htr::Point3D(1, -1, 0),
		htr::Point3D(1, 0, 1), htr::Point3D(1, 0, -1), htr::Point3D(-1, 1, 0), htr::Point3D(-1, -1, 0),
		htr::Point3D(-1, 0, 1), htr::Point3D(-1, 0, -1), htr::Point3D(0, 1, 1), htr::Point3D(0, 1, -1),
		htr::Point3D(0, -1, 1), htr::Point3D(0, -1, -1)
	};

	const float latticeWeights[19] =
	{
		1.f / 9.f,
		1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
		1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
		1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
	};

	class LB_Octree
	{
	public:
		LB_Octree(vector<OctreeGenerator::Voxel>	&_octreeVoxels, float _tau);
		~LB_Octree(){};

		void step();

	private:

		float							ro, rovx, rovy, rovz, v_sq_term;
		float							tau, c;
		const int						stride = 19;
		vector<OctreeGenerator::Voxel>	octreeVoxels;

		void							stream();
	};
}


#endif