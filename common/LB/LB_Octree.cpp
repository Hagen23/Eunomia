#include "LB/LB_Octree.h"

namespace htr
{
	LB_Octree::LB_Octree(vector<OctreeGenerator::Voxel>	&_octreeVoxels, float _tau):
		octreeVoxels(_octreeVoxels), tau(_tau)
	{
		ro = rovx = rovy = rovz = v_sq_term = 0;
		c = 1.0 / sqrt(3.0);
	}

	void LB_Octree::step()
	{
		stream();
	}

	void LB_Octree::stream()
	{
		for (OctreeGenerator::Voxel voxel : octreeVoxels)
		{
			if (voxel.type == htr::OctreeGenerator::Voxel::voxel_type::inside)
			{
				if (voxel.neighbors.size() < 19)
					int test = 0;

				for (OctreeGenerator::Voxel *voxelN : voxel.neighbors)
				{
					htr::Point3D direction(
						voxelN->position.x - voxel.position.x,
						voxelN->position.y - voxel.position.y,
						voxelN->position.z - voxel.position.z);

					direction /= voxel.size;

					int speedIndex = std::distance(speedDirection, std::find(speedDirection, speedDirection + 19, direction));
					voxel.ftemp[speedIndex] = voxelN->f[speedIndex];
				}
			}
		}
	}
}