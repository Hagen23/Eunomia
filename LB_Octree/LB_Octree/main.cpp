#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <GL/glut.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <stdio.h>
#include <chrono>

#include <mouseUtils.h>
#include <HTRBasicDataStructures.h>
#include <OctreeGenerator.h>
#include <LB/LB_Octree.h>

#include "AddVertices.h"

#define OCTREE_VOXEL_SIZE 4

using namespace std;

static htr::Point3D speedDirection[19] =
{
	htr::Point3D(0, 0, 0),
	htr::Point3D(1, 0, 0), htr::Point3D(-1, 0, 0), htr::Point3D(0, 1, 0), htr::Point3D(0, -1, 0),
	htr::Point3D(0, 0, 1), htr::Point3D(0, 0, -1), htr::Point3D(1, 1, 0), htr::Point3D(1, -1, 0),
	htr::Point3D(1, 0, 1), htr::Point3D(1, 0, -1), htr::Point3D(-1, 1, 0), htr::Point3D(-1, -1, 0),
	htr::Point3D(-1, 0, 1), htr::Point3D(-1, 0, -1), htr::Point3D(0, 1, 1), htr::Point3D(0, 1, -1),
	htr::Point3D(0, -1, 1), htr::Point3D(0, -1, -1)
};

vector<htr::Point3D>					groupA, gA_centroids, gA_added_points;
AddVertices::centroidMap				points_by_centroid;
htr::Point3D							points_centroid;
htr::OctreeGenerator					*octreeGen;
AddVertices								*added_vertices;
unsigned int							depth_level;

htr::OctreeGenerator::AlignedPointTVector voxelCenters;
vector<htr::OctreeGenerator::Voxel>		voxelNeighbors;

std::chrono::time_point<std::chrono::system_clock> previous_time, current_time;

float cameraDistance = 1;

char fps_message[50];
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

bool sortVoxelByY(const htr::OctreeGenerator::Voxel &a, const htr::OctreeGenerator::Voxel &b) { return a.position.y < b.position.y; };

void calculateFPS()
{
	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
}

void readCloudFromFile(const char* filename, vector<htr::Point3D>* points)
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
		if (aux == 3)
			points->push_back(htr::Point3D(x, y, z, htr::Point3D::point_type::boundary));
	}
}

void init(void)
{
	glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
	glShadeModel(GL_FLAT);
	glEnable(GL_DEPTH_TEST);

	octreeGen = new htr::OctreeGenerator();

	mouseUtils::init(cameraDistance);

	//readCloudFromFile("../Resources/susane.csv", &groupA);
	readCloudFromFile("../Resources/shpere.csv", &groupA);

	for (auto& point : groupA)
		point.scalePoint(20);

	added_vertices = new AddVertices(groupA, 0.5f);

	octreeGen->initCloudFromVector(added_vertices->get_all());
	/*octreeGen->initCloudFromVector<htr::Point3D>(added_vertices->get_added());*/
	octreeGen->initOctree(OCTREE_VOXEL_SIZE);

	printf("\n Octree depth: %d\n", depth_level = octreeGen->getOctreeDepth());

	printf("\n Number of voxels: %d\n", octreeGen->getVoxels().size());

	//htr::OctreeGenerator::Voxel* searchVoxel = &octreeGen->getVoxels().at(500);

	octreeGen->obtainNeighbors(speedDirection, 19);

	htr::LB_Octree LB_OC(octreeGen->getVoxels(), 1.0f);

	//LB_OC.step();

	voxelCenters.clear();

	//for (int i = 0; i < 19; i++)
	//{
	//	pcl::PointXYZ target_direction(
	//		speedDirection[i].x * OCTREE_VOXEL_SIZE,
	//		speedDirection[i].y * OCTREE_VOXEL_SIZE,
	//		speedDirection[i].z * OCTREE_VOXEL_SIZE);

	//	pcl::PointXYZ target_voxel_center(
	//		searchVoxel->position.x + target_direction.x,
	//		searchVoxel->position.y + target_direction.y,
	//		searchVoxel->position.z + target_direction.z);

	//	vector<htr::OctreeGenerator::Voxel>::iterator it;
	//	htr::OctreeGenerator::Voxel auxVoxel;
	//	auxVoxel.position = target_voxel_center;

	//	clock_t begin = clock();

	//	//octreeGen->obtainNeighbors(
	//	//	Eigen::Vector3f(searchVoxel->position.x, searchVoxel->position.y, searchVoxel->position.z),
	//	//	Eigen::Vector3f(speedDirection[i].x, speedDirection[i].y, speedDirection[i].z),
	//	//	voxelCenters, 0);

	//	it = find(octreeGen->getVoxels().begin(), octreeGen->getVoxels().end(), auxVoxel);

	//	clock_t end = clock();
	//	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	//	printf("Find neighbors (%d): %f ms\n", i, elapsed_secs * 1000);

	//	if (it != octreeGen->getVoxels().end())
	//		searchVoxel->neighbors.push_back(&*it);

	//for (auto voxel_center : voxelCenters)
	//{
	//	pcl::PointXYZ error(
	//		(voxel_center.x - searchVoxel->position.x) / OCTREE_VOXEL_SIZE,
	//		(voxel_center.y - searchVoxel->position.y) / OCTREE_VOXEL_SIZE,
	//		(voxel_center.z - searchVoxel->position.z) / OCTREE_VOXEL_SIZE);

	//	if (error.x == speedDirection[i].x && error.y == speedDirection[i].y && error.z == speedDirection[i].z)
	//	{
	//		htr::OctreeGenerator::Voxel *aux_voxel = new htr::OctreeGenerator::Voxel();
	//		aux_voxel->position = voxel_center;
	//		aux_voxel->size = OCTREE_VOXEL_SIZE;
	//		searchVoxel->neighbors.push_back(aux_voxel);
	//		//voxelNeighbors.push_back(aux_voxel);
	//	}
	//}
	//}

	//std::sort(octreeGen->getVoxels().begin(), octreeGen->getVoxels().end(), sortVoxelByY);

	//printf("Group A size %f\n", groupA.size());

	gluLookAt(0, 0, -5, points_centroid.x, points_centroid.y, points_centroid.z, 0, 1, 0);

	previous_time = std::chrono::system_clock::now();
}

void displayVoxelsPoints()
{
	//htr::OctreeGenerator::Voxel point = octreeGen->getVoxels().at(500);

	for (auto point : octreeGen->getCentroids())
	{
		//if (point.y > -6.0f && point.y < 2.0f && point.x < 20.0f && point.x > 12.0f)
		{
			glPushMatrix();
			glPointSize(5);
			glColor3f(1, 0.5, 0);
			glBegin(GL_POINTS);
			glVertex3f(point.x, point.y, point.z);
			//glVertex3f(point.position.x, point.position.y, point.position.z);
			glEnd();
			glPopMatrix();
		}
	}

	//for (auto voxel: voxelCenters)
	//{
	//	glPushMatrix();
	//		glColor3f(1, 1, 0);
	//		glBegin(GL_POINTS);
	//		glVertex3f(voxel.x, voxel.y, voxel.z);
	//		glEnd();
	//	glPopMatrix();
	//}

	//for (auto voxel : point.neighbors)
	///*for (auto voxel : voxelNeighbors)*/
	//{
	//	glPushMatrix();
	//		glPointSize(3);
	//		glColor3f(1, 1, 0);
	//		glBegin(GL_POINTS);
	//		glVertex3f((*voxel).position.x, (*voxel).position.y, (*voxel).position.z);
	//		glEnd();
	//	glPopMatrix();
	//}

	//for (htr::OctreeGenerator::Voxel voxel : octreeGen->getVoxels())
	//{
	//	if (voxel.position.y > 0.0f && voxel.position.y < 1.0f)
	//	{
	//		glBegin(GL_POINTS);
	//		glVertex3f(voxel.position.x, voxel.position.y, voxel.position.z);
	//		glEnd();
	//	}
	//}
}

void displayVoxles()
{
	//htr::OctreeGenerator::Voxel voxel = octreeGen->getVoxels().at(500);
	for (htr::OctreeGenerator::Voxel voxel : octreeGen->getVoxels())
	{
		//if (voxel.position.y > -6.0f && voxel.position.y < -2.0f)
		{
			glPushMatrix();

			if (voxel.type == htr::OctreeGenerator::Voxel::voxel_type::boundary)
				glColor3f(0.0, 1.0, 1.0);
			else
				glColor3f(0.0, 1.0, 0.0);

			glTranslatef(voxel.position.x, voxel.position.y, voxel.position.z);
			glutWireCube(voxel.size);
			glPopMatrix();
		}
	}

	//for (auto voxel : voxelCenters)
	//{
	//	glPushMatrix();
	//		glColor3f(1.0, 1.0, 1.0);
	//		glTranslatef(voxel.x, voxel.y, voxel.z);
	//		glutWireCube(3);
	//	glPopMatrix();
	//}

	//for (auto voxelN : voxel.neighbors)
	////for (auto voxelN : voxelNeighbors)
	//{
	//	glPushMatrix();
	//	glColor3f(1.0, 1.0, 1.0);
	//	glTranslatef((*voxelN).position.x, (*voxelN).position.y, (*voxelN).position.z);
	//	glutWireCube(2.8);
	//	glPopMatrix();
	//}

	//for (int i = 0; i < 19; i++)
	//{
	//	pcl::PointXYZ direction(
	//		speedDirection[i].x * OCTREE_VOXEL_SIZE,
	//		speedDirection[i].y * OCTREE_VOXEL_SIZE,
	//		speedDirection[i].z * OCTREE_VOXEL_SIZE);
	//		
	//	glPushMatrix();
	//		glLineWidth(3);
	//		glColor3f(1.0, 0.0, 1.0);
	//		glBegin(GL_LINES);
	//		glVertex3f(voxel.position.x, voxel.position.y, voxel.position.z);
	//		glVertex3f(voxel.position.x + direction.x, voxel.position.y + direction.y,
	//			voxel.position.z + direction.z);
	//		glEnd();
	//	glPopMatrix();
	//	
	//}
}

void displayPoints()
{
	// Displays de points in the found clusters
	int j = 0;
	for (htr::Point3D point : added_vertices->get_original())
	{
		glColor3f(1.0, 0.0, 0.0);

		glPointSize(3);
		glBegin(GL_POINTS);
		glVertex3f(point.x, point.y, point.z);
		glEnd();
	}

	//for (htr::Point3D point : added_vertices->get_added())
	//{
	//	glColor3f(0.0, 1.0, 0.0);

	//	glPointSize(3);
	//	glBegin(GL_POINTS);
	//		glVertex3f(point.x, point.y, point.z);
	//	glEnd();
	//}

	// Displays the cloud and clusters centroids
	glColor3f(0, 1, 1);

	glPointSize(5);
	glBegin(GL_POINTS);
	glVertex3f(added_vertices->get_centroid().x, added_vertices->get_centroid().y, added_vertices->get_centroid().z);
	glEnd();

	//glColor3f(1, 1, 0);

	//for (centroidMap::iterator it = points_by_centroid.begin(); it != points_by_centroid.end(); ++it)
	//{
	//	glBegin(GL_POINTS);
	//		glVertex3f(it->first.x, it->first.y, it->first.z);
	//	glEnd();
	//}
	//for (htr::Point3D point : gA_centroids)
	//{
	//	glBegin(GL_POINTS);
	//		glVertex3f(point.x, point.y, point.z);
	//	glEnd();
	//}
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	mouseUtils::applyMouseTransform(points_centroid.x, points_centroid.y, points_centroid.z);
	displayPoints();
	displayVoxelsPoints();
	displayVoxles();
	glPopMatrix();

	glutSwapBuffers();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90.0, (GLfloat)w / (GLfloat)h, 0.001, 10000.0);
	glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27: // ESCAPE
		exit(0);
		break;
	case 'a':
		depth_level++;
		printf("\n Obtaining depth %d...", depth_level);
		octreeGen->extractPointsAtLevel(depth_level);
		break;
	case 's':
		depth_level--;
		printf("\n Obtaining depth %d...", depth_level);
		octreeGen->extractPointsAtLevel(depth_level);
		break;
	default:
		break;
	}
}

void idle()
{
	calculateFPS();
	sprintf(fps_message, "FPS %.3f", fps);
	glutSetWindowTitle(fps_message);
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouseUtils::mouse);
	glutMotionFunc(mouseUtils::mouseMotion);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	glutMainLoop();
	return 0;
}
