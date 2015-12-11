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
#include <ShapeMatching/deformable.h>

#include <AddVertices.h>

using namespace std;

// Map definition to store a given number of points for a determined centroid
typedef map<htr::Point3D, vector<htr::Point3D>> centroidMap;

vector<htr::Point3D>	groupA, gA_centroids, gA_added_points;
centroidMap				points_by_centroid;
htr::Point3D			points_centroid; 
htr::OctreeGenerator    *octreeGen;
AddVertices				*added_vertices;
Deformable				*deformableObject;

std::chrono::time_point<std::chrono::system_clock> previous_time, current_time;

m3Vector	boundariesCenter(0, 0, 0);
m3Real		boundariesLength = 2;

float cameraDistance = 1;

char fps_message[50];

int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

bool sortVoxelByY(const htr::OctreeGenerator::Voxel &a, const htr::OctreeGenerator::Voxel &b) { return a.position.y < b.position.y; };

bool sortByY(htr::Point3D a, htr::Point3D b){ return a.y < b.y; };

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
			points->push_back(htr::Point3D(x,y,z));
	}
}

void init(void)
{
	glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
	glShadeModel(GL_FLAT);
	glEnable(GL_DEPTH_TEST);

	octreeGen = new htr::OctreeGenerator();
	deformableObject = new Deformable();

	mouseUtils::init(cameraDistance);

	//readCloudFromFile("../Resources/susane.csv", &groupA);
	readCloudFromFile("../Resources/shpere.csv", &groupA);
	
	added_vertices = new AddVertices(groupA, 0.30f);
	points_centroid = added_vertices->get_centroid();

	for (auto& point : added_vertices->get_all())
		deformableObject->addVertex(m3Vector(point.x, point.y, point.z), 0.1f);
	
	deformableObject->params.bounds.max = m3Vector(boundariesLength, boundariesLength, boundariesLength);
	deformableObject->params.bounds.min = m3Vector(-boundariesLength, -boundariesLength, -boundariesLength);
	deformableObject->params.timeStep = 0.02f;

	octreeGen->initCloudFromVector<htr::Point3D>(added_vertices->get_added());
	octreeGen->initOctree(3);

	std::sort(octreeGen->getVoxels().begin(), octreeGen->getVoxels().end(), sortVoxelByY);

	printf("Group A size %f\n", groupA.size());

	gluLookAt(0, 0, -5, points_centroid.x, points_centroid.y, points_centroid.z, 0, 1, 0);

	previous_time = std::chrono::system_clock::now();
}

void displayDeformable()
{
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glTranslatef(boundariesCenter.x, boundariesCenter.y, boundariesCenter.z);
		glScalef(boundariesLength, boundariesLength, boundariesLength);
		glutWireCube(boundariesLength);
	glPopMatrix();

	//Goal configuration
	glColor3f(0.0, 1.0, 0.0);
	for (int i = 0; i < deformableObject->getNumVertices(); i++)
	{
		m3Vector point = deformableObject->getGoalVertexPos(i);
		glPointSize(3);
		glBegin(GL_POINTS);
		glVertex3f(point.x, point.y, point.z);
		glEnd();
	}

	//Modified Points
	glColor3f(1.0, 0.0, 0.0);
	for (int i = 0; i < deformableObject->getNumVertices(); i++)
	{
		m3Vector point = deformableObject->getVertexPos(i);
		glPointSize(3);
		glBegin(GL_POINTS);
		glVertex3f(point.x, point.y, point.z);
		glEnd();
	}
}

void displayVoxelsPoints()
{
	glPointSize(3);
	for (pcl::PointXYZ point : octreeGen->getCentroids())
	{
		if (point.y > -6.0f && point.y < -2.0f)
		{
			glBegin(GL_POINTS);
			glVertex3f(point.x, point.y, point.z);
			glEnd();
		}
	}
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
	for (htr::OctreeGenerator::Voxel voxel : octreeGen->getVoxels())
	{
		if (voxel.position.y > -6.0f && voxel.position.y < -2.0f)
		{
			glPushMatrix();
				glColor3f(1.0, 1.0, 1.0);
				glTranslatef(voxel.position.x, voxel.position.y, voxel.position.z);
				glutWireCube(voxel.size);
			glPopMatrix();
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

	glColor3f(1, 1, 0);

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
	//displayPoints();
	/*displayVoxelsPoints();
	displayVoxles();*/
	displayDeformable();
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
		deformableObject->params.alpha += 0.01;
		break;
	case 's':
		deformableObject->params.alpha -= 0.01;
		break;
	case 'v':
		deformableObject->params.volumeConservation = !deformableObject->params.volumeConservation;
		break;
	case 'q':
		deformableObject->params.quadraticMatch = !deformableObject->params.quadraticMatch;
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
	deformableObject->timeStep();
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
