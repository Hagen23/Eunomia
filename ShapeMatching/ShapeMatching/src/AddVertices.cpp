#include "AddVertices.h"


AddVertices::AddVertices()
{
}

AddVertices::AddVertices(vector<htr::Point3D> inputPoints, float _distance_between_points) :
originalPoints(inputPoints), distance_between_points(_distance_between_points)
{
	sort(originalPoints.begin(), originalPoints.end(), &AddVertices::sortByY);
	calculateCentroid();
	createVertices();
}

AddVertices::~AddVertices()
{
}


void AddVertices::createVertices()
{
	obtainCentroidsByLevel(originalPoints, points_centroids, points_by_centroid);
	addPointsByLevel(points_by_centroid, added_points);
	all_points.reserve(originalPoints.size() + added_points.size());
	all_points.insert(all_points.end(), originalPoints.begin(), originalPoints.end());
	all_points.insert(all_points.end(), added_points.begin(), added_points.end());
}

void AddVertices::calculateCentroid()
{
	if (originalPoints.size() > 0)
	{
		for (htr::Point3D point : originalPoints)
			originalPoints_centroid += point;

		originalPoints_centroid /= originalPoints.size();
	}
	else
		originalPoints_centroid = htr::Point3D();
}

void AddVertices::obtainCentroidsByLevel(vector<htr::Point3D> sorted_points, vector<htr::Point3D>& points_centroids,
	AddVertices::centroidMap& centroid_points)
{
	float current_Y = sorted_points.begin()._Ptr->y;
	htr::Point3D* current_centroid = new htr::Point3D();
	vector<htr::Point3D>* points = new vector<htr::Point3D>();
	unsigned int size = 0;

	//Group all the points that have the same y, with their calculated centroid
	for (htr::Point3D point : sorted_points)
	{
		if (point.y != current_Y)
		{
			*current_centroid /= size;
			points_centroids.push_back(*current_centroid);

			std::pair<centroidMap::iterator, bool> res =
				centroid_points.insert(make_pair(*current_centroid, *points));
			if (!res.second)
				cout << "key already exists with value ";

			current_centroid = new htr::Point3D();
			points->clear();
			points = new vector<htr::Point3D>();
			current_Y = point.y;
			size = 0;
		}
		*current_centroid += point;
		points->push_back(point);
		size++;
	}

	delete current_centroid;
	delete points;
}


void AddVertices::addPointsByLevel(centroidMap& points_centroids, vector<htr::Point3D>& added_points)
{
	float distance_centroid_point, points_per_segment;

	for (centroidMap::iterator it = points_centroids.begin(); it != points_centroids.end(); ++it)
	{
		for (auto& point : it->second)
		{
			distance_centroid_point = sqrtf(pow(point.x - it->first.x, 2) + (pow(point.z - it->first.z, 2)));
			points_per_segment = distance_centroid_point / distance_between_points;

			for (float i = 1; i < points_per_segment; i++)
			{
				htr::Point3D aux = createPointbyLevel(it->first, point, i*distance_between_points);
				if (aux.x != it->first.x || aux.y != it->first.y || aux.z != it->first.z)
					added_points.push_back(aux);
			}
		}
	}
}

htr::Point3D AddVertices::createPointbyLevel(htr::Point3D centroid, htr::Point3D point, float scale)
{
	htr::Point3D aux;

	//Distance from the centroid to the bounday point
	float distance_centroid_point = sqrtf(pow(point.x - centroid.x, 2) + (pow(point.z - centroid.z, 2)));

	//Create a vector from the centroid and the point given
	htr::Point3D direction(point.x - centroid.x, point.y - centroid.y, point.z - centroid.z);
	float magnitude = sqrtf(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);

	//Normalize the vector with its magnitude, and scale it to the distance wanted for the new point
	direction /= magnitude;
	direction *= scale;

	//Add the reference vector to the centroid
	aux = centroid + direction;

	float distance_centroid_newPoint = sqrtf(pow(aux.x - centroid.x, 2) + (pow(aux.z - centroid.z, 2)));

	//If the distance from the centroid to the new point is larger than from the centroid to the boundary point, 
	//the centroid is returned -- This is done to avoid adding points outside the mesh
	if (distance_centroid_newPoint < distance_centroid_point)
		return aux;
	else return centroid;
}
