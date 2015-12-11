/**
*@class OctreeGenerator
*Creates an octree from the point cloud data provided.
*
* @author  Jonathan Langford
*
*
*/

#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#pragma once
#ifndef OCTREE_GENERATOR_H
#define OCTREE_GENERATOR_H

//#include <pcl/point_cloud.h>
//#include <pcl/octree/octree.h>
#include "HTRBasicDataStructures.h"
#include <pcl/octree/octree_impl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <ThreadPool.h>
#include <thread>

using std::vector;
using std::string;

namespace htr
{
	class OctreeGenerator{

	public:

		typedef pcl::PointCloud<pcl::PointXYZ>											CloudXYZ;
		typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>						OctreeXYZSearch;
		typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNode			LeafNode;
		typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNodeIterator	LeafNodeIterator;
		typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector AlignedPointTVector;

		/// A voxel that will contain the LB information
		struct Voxel{

			///Each voxel will be either inside a boundary (defined by the original mesh), a boundary, or air surrounding the voxels
			enum voxel_type { inside, boundary, air};

			pcl::PointXYZ position;
			float size;
			voxel_type type;

			///Contains all the neighbors of a voxel. Will be used for the LB stream step
			vector<Voxel*> neighbors;

			/// Necessary for the LB implementation. May remove this in a later stage.
			float *f, *ftemp, *feq;
			pcl::PointXYZ velocityVector;

			///The voxel's LB model is the D3Q19, and the distributions will be initialized as such.
			Voxel(const pcl::PointXYZ &_position, float _size, voxel_type _type):
				position(_position), size(_size), type(_type)
			{
				f = new float[19];
				ftemp = new float[19];
				feq = new float[19];
				for (int i = 0; i < 19; i++)
				{
					f[i] = 0; 
					ftemp[i] = 0;
					feq[i] = 0;
				}
				velocityVector = pcl::PointXYZ(0, 0, 0);
			}

			~Voxel()
			{
			}

			///Had to compare to a defined error, because operations on floats could give different results:
			///5.799 instead of 5.8, for example.
			bool operator== (const Voxel &v) const
			{
				return (
					fabs(this->position.x - v.position.x) < 0.001f &&
					fabs(this->position.y - v.position.y) < 0.001f &&
					fabs(this->position.z - v.position.z) < 0.001f);
			}
			
			//http://stackoverflow.com/questions/28488986/c-opengl-formal-parameter-with-declspecalign16-wont-be-aligned
			static void* operator new(std::size_t sz){
				return ::operator new(sz);
			}

			static void operator delete(void* ptr)
			{
				::operator delete(ptr);
			}
		};

		OctreeGenerator();
		~OctreeGenerator();

		inline CloudXYZ::Ptr			getCloud(){return cloud;}
		inline pcl::PointXYZ			getCloudCentroid(){return cloudCentroid;}
		inline vector<Voxel>&			getVoxels(){return octreeVoxels;}
		inline vector<pcl::PointXYZ>&	getCentroids(){ return octreeCentroids; }
		inline OctreeXYZSearch::Ptr		getOctree() {return octree_p;}
		inline unsigned int				getOctreeDepth(){ return octree_p->getTreeDepth(); }

		///Finds all the voxel's neighbors. The directions where this method looks are based on the LB D3Q19 model.
		///@param[in] directions The array that contains the D3Q19 directions.
		///@param[in] directions_size The size of the direction array.
		void obtainNeighbors(htr::Point3D *directions, int directions_size);

		///Initializes pcl's cloud data structure with random values centered at 0,0,0.
		///@param[in] width The width of the point cloud.
		///@param[in] height The height of the point cloud.
		///@param[in] depth The depth of the point cloud.
		///@param[in] numOfPoints The num of points in the point cloud.
		void initRandomCloud(const float width,const float height,const float depth,const int numOfPoints);

		///Initializes pcl's cloud data structure from a point cloud file stored as comma separated values.
		void initCloudFromFile(string fileName);
		
		///Initializes pcl's cloud data structure with values from a file.
		///@param[in] filename  The location of the file to open
        void readCloudFromFile(const char* filename);
		
		///Initializes pcl's cloud data structure from a vector of any type containing x, y, and z member variables.
		///@param[in] points The input data vector.
		template <typename T>
		void initCloudFromVector(const vector<T>& points);
		
		///Initializes pcl's cloud data structure from a vector of any type containing x, y, and z member variables.
		///@param[in] points The input data vector.
		void initCloudFromVector(const vector<Point3D>& points);
		
		///Initializes the octree from the cloud data provided at the specified resolution.
		///@param[in] resolution The voxel edge size at the minimum subdivision level.
		void initOctree(const int resolution);
		
		///Calculates the position of each voxel that exists at a specified tree depth.
		///@param[in] depth The selected tree depth in the octree.
		void extractPointsAtLevel(const int depth);

		//http://en.cppreference.com/w/cpp/memory/new/operator_new
		static void* operator new(std::size_t sz){
			return ::operator new(sz);
		}
		// custom placement delete
		static void operator delete(void* ptr)
		{
			::operator delete(ptr);
		}

	private:
		
		std::mutex				mutex;
		unsigned int			currentExtractionLevel;
		CloudXYZ::Ptr			cloud;
		OctreeXYZSearch::Ptr	octree_p;
		pcl::PointXYZ			cloudCentroid;
		vector<Voxel>			octreeVoxels;
		vector<pcl::PointXYZ>	octreeCentroids;
		vector<int>				boundary_point_indexes;

		///Calculates the entire cloud centroid
		void calculateCloudCentroid();
	};

	template <typename T>
	void OctreeGenerator::initCloudFromVector(const vector<T>& points){
		//Note: Width and Height are only used to store the cloud as an image.
		//Source width and height can be used instead of a linear representation.
		cloud->width = points.size();
		cloud->height = 1;

		cloud->points.resize(cloud->width * cloud->height);

		for (size_t i = 0; i < cloud->points.size (); ++i){
			cloud->points[i].x = points[i].x;
			cloud->points[i].y = points[i].y;
			cloud->points[i].z = points[i].z;
		}
        calculateCloudCentroid();
	}

}

#endif
