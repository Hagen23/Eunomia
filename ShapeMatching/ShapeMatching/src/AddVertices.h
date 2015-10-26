#pragma once
#ifndef ADD_VERTICES
#define ADD_VERTICES

#include <OctreeGenerator.h>
#include <HTRBasicDataStructures.h>
#include <vector>
#include <map>

using namespace std;

class AddVertices
{
	public:

		/// Map definition to store a given number of points for a determined centroid
		typedef map<htr::Point3D, vector<htr::Point3D>> centroidMap;

		//In the first case cmp is declared as a member function of the class sample and hence 
		//requires this pointer for calling it. Since the this pointer is not available compiler 
		//is complaining about it.You can make it work by declaring cmp as static function since 
		//static functions do not require this pointer for calling
		static bool sortByY(htr::Point3D a, htr::Point3D b){ return a.y < b.y; };

		AddVertices();
		AddVertices(vector<htr::Point3D> inputPoints, float _distance_between_points);

		~AddVertices();

		htr::Point3D get_centroid(){ return originalPoints_centroid; }
		vector<htr::Point3D> get_original(){ return originalPoints; }
		vector<htr::Point3D> get_added(){ return added_points; }
		vector<htr::Point3D> get_all(){ return all_points; }
		
	private:
		
		htr::Point3D			originalPoints_centroid;
		vector<htr::Point3D>	originalPoints, points_centroids, added_points, all_points;
		centroidMap				points_by_centroid;
		float					distance_between_points;
		
		///Calculates the centroid of the input points
		void calculateCentroid();

		///Creates the additional vertices
		void createVertices();

		///Assigns the points of a mesh to a given centroid
		///@param[in] sorted_points		The mesh points that have been sorted over an Axis; y in this case
		///@param[in] points_centroids	A vector that will contain all the calculated centroids -- May not be needed anymore
		///@param[in] centroid_points	The map that will contain the calculated centroids and the points that were used to calculate them
		void obtainCentroidsByLevel(vector<htr::Point3D> sorted_points, vector<htr::Point3D>& points_centroids,
			AddVertices::centroidMap& centroid_points);

		///For a given map, creates a new set of points that reside inside the point mesh
		///@param[in] points_centroids	The map that contains the centroids and the points related to the centroid
		///@param[in] added_points		The vector that will hold the new points
		///TODO: automatically calculate the number of iterations in order to add all the points
		void addPointsByLevel(centroidMap& points_centroids, vector<htr::Point3D>& added_points);

		///Creates a new point given a centroid and a boundary point
		///@param[in] centroid	The reference centroid
		///@param[in] point		The boundary point
		///@param[in] scale		The distance from the centroid to the new point
		htr::Point3D createPointbyLevel(htr::Point3D centroid, htr::Point3D point, float scale);
};

#endif