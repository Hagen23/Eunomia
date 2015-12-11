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

#include "AddVertices.h"

#define OCTREE_VOXEL_SIZE 4

using namespace std;

static htr::Point3D speedDirection[19] =
{
	htr::Point3D( 0, 0, 0 ),
	htr::Point3D( 1, 0, 0 ), htr::Point3D( -1, 0, 0 ), htr::Point3D( 0, 1, 0 ), htr::Point3D( 0, -1, 0 ),
	htr::Point3D( 0, 0, 1 ), htr::Point3D( 0, 0, -1 ), htr::Point3D( 1, 1, 0 ), htr::Point3D( 1, -1, 0 ),
	htr::Point3D( 1, 0, 1 ), htr::Point3D( 1, 0, -1 ), htr::Point3D( -1, 1, 0 ), htr::Point3D( -1, -1, 0 ),
	htr::Point3D( -1, 0, 1 ), htr::Point3D( -1, 0, -1 ), htr::Point3D( 0, 1, 1 ), htr::Point3D( 0, 1, -1 ),
	htr::Point3D( 0, -1, 1 ), htr::Point3D( 0, -1, -1 )
};

vector<htr::Point3D>					groupA, gA_centroids, gA_added_points;
AddVertices::centroidMap				points_by_centroid;
htr::Point3D							points_centroid; 
htr::OctreeGenerator					*octreeGen;
AddVertices								*added_vertices;
unsigned int							depth_level;

std::chrono::time_point<std::chrono::system_clock> previous_time, current_time;

float cameraDistance = 1;

char fps_message[50];
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

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
			points->push_back(htr::Point3D(x,y,z, htr::Point3D::point_type::boundary));
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

	for (auto& point :groupA)
			point.scalePoint(20);

	vector<htr::Point3D> groupB;

	for (int i = 0; i < 50; i++)
		groupB.push_back(groupA.at(i));
	
	clock_t begin = clock();
	added_vertices = new AddVertices(groupA, 1.5f);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	printf("Add vertices %f ms\n", elapsed_secs * 1000);

	octreeGen->initCloudFromVector(added_vertices->get_all());

	octreeGen->initOctree(OCTREE_VOXEL_SIZE);

	printf("Octree depth: %d\n", depth_level = octreeGen->getOctreeDepth());

	printf("Number of voxels: %d\n", octreeGen->getVoxels().size());

	octreeGen->obtainNeighbors(speedDirection, 19);

	gluLookAt(0, 0, -5, points_centroid.x, points_centroid.y, points_centroid.z, 0, 1, 0);

	previous_time = std::chrono::system_clock::now();
}

void displayVoxelsPoints()
{
	for (auto point : octreeGen->getCentroids())
	{
		glPushMatrix();
			glPointSize(5);
			glColor3f(1, 0.5, 0);
			glBegin(GL_POINTS);
			glVertex3f(point.x, point.y, point.z);
			glEnd();
		glPopMatrix();
	}
}

void displayVoxles()
{
	int air_counter = 0;
	
	for (htr::OctreeGenerator::Voxel voxel : octreeGen->getVoxels())
	{

		glPushMatrix();

		glLineWidth(3);
		if (voxel.type == htr::OctreeGenerator::Voxel::voxel_type::boundary)
		{
			//glPopMatrix();
			//continue;
			glColor3f(0.0, 1.0, 1.0);
		}
		else if (voxel.type == htr::OctreeGenerator::Voxel::voxel_type::inside)
		{
			//glPopMatrix();
			//continue;
			glColor3f(0.0, 1.0, 0.0);
		}
		else if (voxel.type == htr::OctreeGenerator::Voxel::voxel_type::air)
			glColor3f(1.0, 1.0, 0.0);

		glTranslatef(voxel.position.x, voxel.position.y, voxel.position.z);
		glutWireCube(voxel.size);
		glPopMatrix();		

		for (htr::OctreeGenerator::Voxel *voxelN : voxel.neighbors)
		{
			if (voxelN->type == htr::OctreeGenerator::Voxel::voxel_type::air)
			{
				glPushMatrix();
				glLineWidth(3);
				glColor3f(1.0, 0.0, 1.0);
				glTranslatef(voxelN->position.x, voxelN->position.y, voxelN->position.z);
				glutWireCube(voxelN->size);
				glPopMatrix();
			}
		}
	}
}

void displayPoints()
{
	// Displays de points in the found clusters
	int j = 0;
	for (htr::Point3D point: added_vertices->get_original())
	{
		glColor3f(1.0,0.0,0.0);

		glPointSize(3);
		glBegin(GL_POINTS);
			glVertex3f(point.x, point.y, point.z);
		glEnd();
	}

	for (htr::Point3D point : added_vertices->get_added())
	{
		glColor3f(0.0, 1.0, 0.0);
		
		if (point.type == htr::OctreeGenerator::Voxel::voxel_type::air)
			glColor3f(1.0, 1.0, 0.0);

		glPointSize(3);
		glBegin(GL_POINTS);
			glVertex3f(point.x, point.y, point.z);
		glEnd();
	}

	// Displays the cloud and clusters centroids
	glColor3f(0, 1, 1);

	glPointSize(5);
	glBegin(GL_POINTS);
	glVertex3f(added_vertices->get_centroid().x, added_vertices->get_centroid().y, added_vertices->get_centroid().z);
	glEnd();
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	mouseUtils::applyMouseTransform(points_centroid.x, points_centroid.y, points_centroid.z);
	//displayPoints();
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
