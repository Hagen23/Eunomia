#include <OctreeGenerator.h>
#include <fstream>
#include <ctime>

using std::ifstream;
using std::iterator;

namespace htr{

	///The default constructor.
	OctreeGenerator::OctreeGenerator():
		cloud (new CloudXYZ),
		octree_p(new OctreeXYZSearch(20)),
		currentExtractionLevel(0)
	{
	}

	///The default destructor.
	OctreeGenerator::~OctreeGenerator(){
	}

	void OctreeGenerator::initCloudFromVector(const vector<Point3D>& points){
		//Note: Width and Height are only used to store the cloud as an image.
		//Source width and height can be used instead of a linear representation.
		cloud->width = points.size();
		cloud->height = 1;

		cloud->points.resize(cloud->width * cloud->height);

		for (size_t i = 0; i < cloud->points.size(); ++i)
		{
			cloud->points[i].x = points[i].x;
			cloud->points[i].y = points[i].y;
			cloud->points[i].z = points[i].z;

			//The index of each boundary point is stored in order to later determine which voxel is a boundary voxel.
			if (points[i].type == Point3D::point_type::boundary)
				boundary_point_indexes.push_back(i);
		}
		calculateCloudCentroid();
	}

	void OctreeGenerator::initRandomCloud(const float width,const float height,const float depth,const int numOfPoints)
	{
		srand ((unsigned int) time (NULL));

		cloud->width = numOfPoints;
		cloud->height = 1;
		cloud->points.resize(cloud->width * cloud->height);

		for (size_t i = 0; i < cloud->points.size (); ++i)
		{
			cloud->points[i].x = (width * rand () / (RAND_MAX + 1.0f)) - width/2;
			cloud->points[i].y = (height * rand () / (RAND_MAX + 1.0f)) - height/2;
			cloud->points[i].z = (depth * rand () / (RAND_MAX + 1.0f)) - depth/2;
		}
	}

    void OctreeGenerator::readCloudFromFile(const char* filename)
    {
        FILE *ifp;
        float x, y, z;
        int aux = 0;

        if ((ifp = fopen(filename, "r")) == NULL)
        {
          fprintf(stderr, "Can't open input file!\n");
          return;
        }

        while ((aux = fscanf(ifp, "%f,%f,%f\n", &x, &y, &z)) != EOF)
        {
            if(aux == 3)
                cloud->points.push_back(pcl::PointXYZ(x, y, z));
        }

        fclose(ifp);

        cloud->width = cloud->points.size();
		cloud->height = 1;

		calculateCloudCentroid();
    }

    void OctreeGenerator::calculateCloudCentroid()
    {
        for(pcl::PointXYZ point:cloud->points)
        {
            cloudCentroid.x+=point.x;
            cloudCentroid.y+=point.y;
            cloudCentroid.z+=point.z;
        }
        cloudCentroid.x/=cloud->points.size();
        cloudCentroid.y/=cloud->points.size();
        cloudCentroid.z/=cloud->points.size();
    }

	void OctreeGenerator::initCloudFromFile(string fileName)
	{
		cloud->points.clear();
		ifstream file (fileName); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
		string value;
		double points[3];
		while ( file.good() )
		{
			for(int i=0;i<3;++i)
			{
				getline ( file, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
				points[i] = atof(value.c_str());
			}
			cloud->points.push_back(pcl::PointXYZ(points[0], points[1], points[2]));
		}

		cloud->width = cloud->points.size();
		cloud->height = 1;
	}

	void OctreeGenerator::initOctree(const int resolution)
	{
		//octree_p.reset(new OctreeXYZSearch(resolution));
		octree_p->deleteTree();
		octree_p->setResolution(resolution);

		octree_p->setInputCloud (cloud);
		octree_p->addPointsFromInputCloud();

		currentExtractionLevel = octree_p->getTreeDepth();
		extractPointsAtLevel(currentExtractionLevel);
	}

	void OctreeGenerator::extractPointsAtLevel(const int depth)
	{
		vector<int> pointIdxVec;
		bool isBoundary = false;
		//This variable will store the voxel's centers.
		AlignedPointTVector voxel_center_list;

		clock_t begin = clock();
		if (depth >= 0 && depth <= octree_p->getTreeDepth())
		{
			currentExtractionLevel = depth;
			octreeVoxels.clear();
			octreeCentroids.clear();

			//Gets the centroids of each voxel that contains a point.
			octree_p->getOccupiedVoxelCenters(voxel_center_list);
			double length = sqrtf(octree_p->getVoxelSquaredSideLen());

			for (auto& voxel_center : voxel_center_list)
			{
				isBoundary = false;
				pointIdxVec.clear();

				pcl::PointXYZ p = {	voxel_center.x, voxel_center.y, voxel_center.z	};

				//Obtains the ids of the points inside a voxel, and checks if said points are boundaries.
				//If they are, the voxel is marked as a boundary.
				if (octree_p->voxelSearch(p, pointIdxVec))
				{
					vector<int>::iterator it;
					for (size_t i = 0; i < pointIdxVec.size(); ++i)
					{
						it = find(boundary_point_indexes.begin(), boundary_point_indexes.end(), pointIdxVec[i]);
						if (it != boundary_point_indexes.end())
						{
							isBoundary = true;
							break;
						}
					}
				}

				Voxel v(p, length,isBoundary ? Voxel::voxel_type::boundary : Voxel::voxel_type::inside);

				octreeVoxels.push_back(v);
				octreeCentroids.push_back(p);
			}

			clock_t end = clock();
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

			printf("Voxel generation %f ms\n", elapsed_secs * 1000);
		}
	}

	void  OctreeGenerator::obtainNeighbors(htr::Point3D *directions, int directions_size)
	{
		//A thread pool is used to speed up the neighbor seach process.
		ThreadPool pool(4);
		clock_t begin = clock();
		vector<Voxel> air_voxels;

		for (int i = 0; i < octreeVoxels.size(); i++)
		{
			//Each voxel is enqueued to the pool, and its neighbors are searched. 
			pool.Enqueue([&, i, this]()
			{
				Voxel* searchVoxel = &octreeVoxels.at(i);

				for (int j = 0; j < directions_size; j++)
				{
					pcl::PointXYZ target_direction(
						directions[j].x * searchVoxel->size,
						directions[j].y * searchVoxel->size,
						directions[j].z * searchVoxel->size);

					pcl::PointXYZ target_voxel_center(
						searchVoxel->position.x + target_direction.x,
						searchVoxel->position.y + target_direction.y,
						searchVoxel->position.z + target_direction.z);

					//We look for existing voxels first, if the voxel does not already exist, it means that the 
					//neighbor is an air voxel.
					vector<htr::OctreeGenerator::Voxel>::iterator it;
					htr::OctreeGenerator::Voxel aux_voxel(target_voxel_center, searchVoxel->size, searchVoxel->type);

					it = find(octreeVoxels.begin(), octreeVoxels.end(), aux_voxel);

					if (it != octreeVoxels.end())
						searchVoxel->neighbors.push_back(&*it);
					else
					{
						Voxel *airVoxel = new Voxel(target_voxel_center, searchVoxel->size, Voxel::voxel_type::air);

						//An optimization could be performed where the air voxels are stored by reference. Currently,
						//if a voxel has an air voxel neighbor, the voxel is added, regardless if it is already the neighbor
						//of another voxel. With the following mutex, this problem is solved, however, it slows down the 
						//process considerably. 

						//lock_guard<std::mutex> guard(this->mutex);
						//{
						//	vector<Voxel>::iterator it_air;
						//	it_air = find(air_voxels.begin(), air_voxels.end(), *airVoxel);

						//	if (it_air == air_voxels.end())
						//	{
						//		air_voxels.push_back(*airVoxel);
								searchVoxel->neighbors.push_back(airVoxel);
						//	}
						//	else
						//		searchVoxel->neighbors.push_back(&*it_air);
						//}
						//this->mutex.unlock();
					}
				}
			});
		}

		//pool.ShutDown();

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		printf("Find neighbors %f ms\n", elapsed_secs * 1000);
	}
}
