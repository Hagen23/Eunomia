#ifndef DBSCAN
#define DBSCAN

#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>

#include "OctreeGenerator.h"
#include "HTRBasicDataStructures.h"
#include "cluster.h"

namespace dbScanSpace
{
   class dbscan
   {
       public:

           dbscan(const char* filename, const int octreeResolution_, const float eps_,
                  const int minPtsAux_, const int minPts_ );

            dbscan();

           ~dbscan();

            template <typename T>
            void init(const vector<T>& points, const int octreeResolution_, const float eps_,
                    const int minPtsAux_, const int minPts_);

			inline vector<htr::OctreeGenerator::Voxel>& getOctreeVoxels() { return octreeGen->getVoxels(); }
           inline vector<cluster>& getClusters(){return clusters;}
           inline htr::OctreeGenerator::CloudXYZ::Ptr getCloudPoints(){return octreeGenIn->getCloud();}
           inline pcl::PointXYZ getCentroid(){return centroid;}

           template <typename T>
           void generateClusters(vector<vector<T>> * clusters);

           void generateClusters();
		   void generateClusters_fast();

       private:

           float                   eps;
           int                     minPtsAux, minPts, octreeResolution;

           vector<cluster>         clustersAux, clusters;
           htr::OctreeGenerator    *octreeGenIn, *octreeGen;
           vector<pcl::PointXYZ>   clustersCentroids;
           pcl::PointXYZ           centroid;

           void calculateCentroid(vector<pcl::PointXYZ> group);
           void octreeRegionQuery(htr::OctreeGenerator *octreeGen, pcl::PointXYZ& searchPoint, double eps, vector<int> *retKeys);
           void DBSCAN_Octree_merge(htr::OctreeGenerator *octreeGen, float eps, int minPts);
           void DBSCAN_Octree(htr::OctreeGenerator *octreeGen, float eps, int minPts);
		   void DBSCAN_Octree_fast(htr::OctreeGenerator *octreeGen, float eps, int minPts);
		   void DBSCAN_Octree_fast2(htr::OctreeGenerator *octreeGen, int minPts);
   };

   ///Initializes the point cloud from a vector, and the octree.
///@param[in] points            Points from which the cloud will be initialized.
///@param[in] octreeResolution_ Resolution with which the octree is initialized.
///@param[in] eps_              The search radius for the octree.
///@param[in] minPtsAux_        Minimum points for the initial clusters.
///@param[in] minPts_           Minimum points for the final clusters.
   template <typename T>
   void dbscan::init(const vector<T>& points, const int octreeResolution_, const float eps_, const int minPtsAux_, const int minPts_)
   {
       eps = eps_;
       minPts = minPts_;
       minPtsAux = minPtsAux_;
       octreeResolution = octreeResolution_;

       octreeGenIn->initCloudFromVector<T>(points);
       octreeGenIn->initOctree(octreeResolution);

       centroid = octreeGenIn->getCloudCentroid();
   }

   template <typename T>
   void dbscan::generateClusters(vector<vector<T>> * clustersOut)
   {
       // A first set of clusters is generated. This first set has a large number of small clusters.
       DBSCAN_Octree(octreeGenIn, eps, minPtsAux);

       // The clusters centroids are calculated and used to generate a second octree.
       for(dbScanSpace::cluster cluster:clustersAux)
           clustersCentroids.push_back(cluster.centroid);

       octreeGen->initCloudFromVector<pcl::PointXYZ>(clustersCentroids);
       octreeGen->initOctree(octreeResolution);

       // Using the second octree and the centroids of the clusters, a new set of clusters is generated.
       // These are the final clusters.
       DBSCAN_Octree_merge(octreeGen, 2*eps, minPts);

//       for(int i = 0; i<clusters.size(); i++)
//           clusters[i].toPoint3D();
   }
}

#endif

//#ifndef DBSCAN
//#define DBSCAN
//
//#include <vector>
//#include <algorithm>
//#include <omp.h>
//#include <cmath>
//
//#include "OctreeGenerator.h"
//#include "HTRBasicDataStructures.h"
//#include "cluster.h"
//
//namespace dbScanSpace
//{
//    class dbscan
//    {
//        public:
//
//            dbscan(const char* filename, const int octreeResolution_, const float eps_,
//                   const int minPtsAux_, const int minPts_ );
//
//            template <typename T>
//            dbscan(const vector<T>& points, const int octreeResolution_, const float eps_,
//                   const int minPtsAux_, const int minPts_);
//
//            ~dbscan();
//
//            inline vector<cluster> getClusters(){return clusters;}
//            inline htr::OctreeGenerator::CloudXYZ::Ptr getCloudPoints(){return octreeGenIn->getCloud();}
//            inline pcl::PointXYZ getCentroid(){return centroid;}
//
//            void generateClusters();
//
//        private:
//
//            float                   eps;
//            int                     minPtsAux, minPts, octreeResolution;
//
//            vector<cluster>         clustersAux, clusters;
//            htr::OctreeGenerator    *octreeGenIn, *octreeGen;
//            vector<pcl::PointXYZ>   clustersCentroids;
//            pcl::PointXYZ           centroid;
//
//            void calculateCentroid(vector<pcl::PointXYZ> group);
//            void octreeRegionQuery(htr::OctreeGenerator *octreeGen, pcl::PointXYZ searchPoint, double eps, vector<int> *retKeys);
//            void DBSCAN_Octree_merge(htr::OctreeGenerator *octreeGen, float eps, int minPts,
//                       vector<cluster> *clustersIn, vector<cluster> *clustersOut);
//            void DBSCAN_Octree(htr::OctreeGenerator *octreeGen, float eps, int minPts, vector<cluster> *clusters);
//    };
//
//    ///Initializes the point cloud from a vector, and the octree.
//	///@param[in] points            Points from which the cloud will be initialized.
//	///@param[in] octreeResolution_ Resolution with which the octree is initialized.
//	///@param[in] eps_              The search radius for the octree.
//	///@param[in] minPtsAux_        Minimum points for the initial clusters.
//	///@param[in] minPts_           Minimum points for the final clusters.
//    template <typename T>
//    dbscan::dbscan(const vector<T>& points, const int octreeResolution_, const float eps_, const int minPtsAux_, const int minPts_)
//    {
//        octreeGenIn = new htr::OctreeGenerator();
//        octreeGen = new htr::OctreeGenerator();
//
//        eps = eps_;
//        minPts = minPts_;
//        minPtsAux = minPtsAux_;
//        octreeResolution = octreeResolution_;
//
//        octreeGenIn->initCloudFromVector<T>(points);
//        octreeGenIn->initOctree(octreeResolution);
//
//        centroid = octreeGenIn->getCloudCentroid();
//    }
//}
//
//#endif
