#pragma once

#define _CRT_SECURE_NO_WARNINGS

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include "lb_src\Lattice.h"

using namespace std;

#define LATTICE_DIM					32

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

float dt = 0;

int latticeWidth = LATTICE_DIM, latticeHeight = LATTICE_DIM, latticeDepth = LATTICE_DIM, ncol;
float latticeViscosity = 15.0f, roIn = 0.1f;
bool withSolid = false, keypressed = false, showInterfase = false, simulate = false;
vector3d vectorIn(0.f, 0.f, 0.f);
latticed3q19 *lattice; 

ofstream logFile;
unsigned int time_step = 0;

float cubeFaces[24][3] = 
{
	{0.0,0.0,0.0}, {1.0,0.0,0.0}, {1.0,1.0,0.0}, {0.0,1.0,0.0},
	{0.0,0.0,0.0}, {1.0,0.0,0.0}, {1.0,0,1.0}, {0.0,0,1.0},
	{0.0,0.0,0.0}, {0,1.0,0.0}, {0,1.0,1.0}, {0.0,0,1.0},
	{0.0,1.0,0.0}, {1.0,1.0,0.0}, {1.0,1.0,1.0}, {0.0,1.0,1.0},
	{1.0,0.0,0.0}, {1.0,1.0,0.0}, {1.0,1.0,1.0}, {1.0,0.0,1.0},
	{0.0,0.0,1.0}, {0,1.0,1.0}, {1.0,1.0,1.0}, {1.0,0,1.0}
};

float getValueFromRelation(float value, float minColorVar=0.01, float maxColorVar=1.0, float minVelVar = -0.01, float maxVelVar = 0.01);

void display (void)
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	int i0;

	float x, y, z, posX, posY, posZ, vx, vy, vz, normMag;

// set view matrix
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	glPushMatrix();
		glColor3f(1.0, 1.0, 1.0);
		glLineWidth(2.0);
		for(int i = 0; i< 6; i++)
		{
			glBegin(GL_LINE_LOOP);
			for(int j = 0; j < 4; j++)
				glVertex3f(cubeFaces[i*4+j][0], cubeFaces[i*4+j][1], cubeFaces[i*4+j][2]);
			glEnd();
		}
	glPopMatrix();

	glPushMatrix();
		for(int k = 0; k  < latticeDepth; k++)
		for(int j = 0; j < latticeHeight; j++)
		for( int i = 0; i < latticeWidth; i++ )
		{
			i0 = I3D(latticeWidth, latticeHeight, i, j, k);

			posX = i / (float)latticeWidth; posY =  j / (float)latticeHeight; posZ = k / (float)latticeDepth;

			if(!lattice->latticeElements[i0].isSolid && ((lattice->latticeElements[i0].cellType & cell_types::fluid) == cell_types::fluid))
			{
				x = getValueFromRelation(lattice->latticeElements[i0].velocityVector.x);
				y = getValueFromRelation(lattice->latticeElements[i0].velocityVector.y);
				z = getValueFromRelation(lattice->latticeElements[i0].velocityVector.z);

				vx = lattice->latticeElements[i0].velocityVector.x;
				vy = lattice->latticeElements[i0].velocityVector.y;
				vz = lattice->latticeElements[i0].velocityVector.z;

				glColor3f(x,y,z);
				normMag = sqrtf(vx*vx + vy*vy + vz*vz)*10; 

				glBegin(GL_LINES);
					glVertex3f(posX, posY, posZ);
					glVertex3f(posX + lattice->latticeElements[i0].velocityVector.x / normMag, posY+ lattice->latticeElements[i0].velocityVector.y / normMag, 
						posZ+ lattice->latticeElements[i0].velocityVector.z / normMag);
				glEnd();
			}
			else if ((lattice->latticeElements[i0].cellType & cell_types::interphase) == cell_types::interphase)
			{
				if (showInterfase)
				{
					glColor3f(1.0, 1.0, 0.0);
					glPointSize(2.0);
					glBegin(GL_POINTS);
					glVertex3f(posX, posY, posZ);
					glEnd();
				}
			}
		}
	glPopMatrix();
		
	// render from the vbo
 //   glBindBuffer(GL_ARRAY_BUFFER, vbo);
 //   glVertexPointer(4, GL_FLOAT, sizeof(Vertex), LOCATION_OFFSET);

	//glColorPointer(4, GL_FLOAT, sizeof(Vertex), COLOR_OFFSET);

 //   glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	//    glDrawArrays(GL_POINTS, 0, DIM * DIM);
	//glDisableClientState(GL_COLOR_ARRAY);
 //   glDisableClientState(GL_VERTEX_ARRAY);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glutSwapBuffers();
}

float getValueFromRelation(float value, float minColorVar, float maxColorVar, float minVelVar , float maxVelVar )
{
	float ratio = (value - minVelVar) / (maxVelVar - minVelVar);
	return ratio * (maxColorVar -  minColorVar) + minColorVar;
}

void idle(void)
{
	if (simulate)
	{
		if (time_step == 29)
			int test = 0;
		//lattice->logInterfaseValues(logFile, std::to_string(time_step));
		lattice->step();
		time_step++;
	}
	
	if(keypressed)
	{
		if(withSolid)
		{
				// Solid inside the cube
			for(int k = latticeDepth/4; k < latticeDepth/2.0 + latticeDepth/4; k++)
				for(int j = latticeHeight/4; j< latticeHeight/2.0 +latticeHeight/4; j++)
					for(int i = latticeWidth/4; i< latticeWidth/2.0 + latticeWidth/4; i++)
					{
						int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
						lattice->latticeElements[i0].isSolid = true;
					}

			keypressed = false;
		}
		else 
		{
				// Solid inside the cube
			for(int k = latticeDepth/4; k < latticeDepth/2.0 + latticeDepth/4; k++)
				for(int j = latticeHeight/4; j< latticeHeight/2.0 +latticeHeight/4; j++)
					for(int i = latticeWidth/4; i< latticeWidth/2.0 + latticeWidth/4; i++)
					{
						int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
						lattice->latticeElements[i0].isSolid = false;
					}

			keypressed = false;
		}
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


void keys (unsigned char key, int x, int y)
{
	static int toonf = 0;
	switch (key) {
		case 27:
			if (logFile.is_open())
				logFile.close();
            exit(0);
			break;
		case 'a':
			keypressed = true;
			withSolid = !withSolid;
			break;
		case 'i':
			showInterfase = !showInterfase;
			break;
		case 's':
			simulate = !simulate;
			break;
	}
}

// Setting up the GL viewport and coordinate system 
void reshape (int w, int h)
{
	glViewport (0,0, w,h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	
	gluPerspective (45, w*1.0/h*1.0, 0.01, 400);
	//glTranslatef (0,0,-5);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
}

// glew (extension loading), OpenGL state and CUDA - OpenGL interoperability initialization
void initGL ()
{
	GLenum err = glewInit();
	if (GLEW_OK != err) 
		return;
	cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << endl;
	const GLubyte* renderer;
	const GLubyte* version;
	const GLubyte* glslVersion;

	renderer = glGetString(GL_RENDERER); /* get renderer string */
	version = glGetString(GL_VERSION); /* version as a string */
	glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
	printf("GLSL version supported %s\n", glslVersion);

	glEnable(GL_DEPTH_TEST);
}

int init(void)
{
	time_t current_time = time(0);
	time(&current_time);
	char* dt = ctime(&current_time);

	cout << "The local date and time is: " << dt << endl;
	logFile.open("Logs/interfaseLog_"+ std::to_string(current_time)+".csv", ios::out | ios::app);
	logFile << dt << endl;
	logFile << "timeStep,Cell Index,NB_Status,ro,cellMass,cellMassTemp,epsilon,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18\n";
	
	int latticeDimension = 16;
	int fluidWidth = latticeDimension, fluidHeight = latticeDimension, fluidDepth = latticeDimension;

	lattice = new latticed3q19(latticeWidth, latticeHeight, latticeDepth, latticeViscosity, 1.0f, LATTICE_DIM, 1.0f);

	for (int k = latticeDepth / 2 - fluidDepth / 2; k < (latticeDepth / 2 + fluidDepth / 2); k++)
	for (int j = latticeHeight / 2 - fluidHeight / 2; j< (latticeHeight / 2 + fluidHeight / 2); j++)
	for (int i = latticeWidth / 2 - fluidWidth / 2; i< (latticeWidth / 2 + fluidWidth / 2); i++)
	{
		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
		lattice->latticeElements[i0].cellType = lattice->latticeElements[i0].cellTypeTemp = cell_types::fluid;
	}

	for (int k = 0; k < (latticeDepth); k++)
	for (int j = 0; j< (latticeHeight); j++)
	for (int i = 0; i < (latticeWidth); i++)
	{
		int cellIndex = I3D(latticeWidth, latticeHeight, i, j, k);
		latticeElementd3q19 *currentCell = &lattice->latticeElements[cellIndex];

		if ((currentCell->cellType & cell_types::fluid) == cell_types::fluid)
		for (int l = 1; l < 19; l++)
		{
			int newI = (int)(i + speedDirection[l].x);
			int newJ = (int)(j + speedDirection[l].y);
			int newK = (int)(k + speedDirection[l].z);

			int neighborIndex = I3D(latticeWidth, latticeHeight, newI, newJ, newK);

			if ((lattice->latticeElements[neighborIndex].cellType & cell_types::gas) == cell_types::gas)
				lattice->latticeElements[neighborIndex].cellType = lattice->latticeElements[neighborIndex].cellTypeTemp
				= cell_types::interphase;
		}
	}

	//for (int k = (latticeDepth / 2 - fluidDepth / 2) - 1; k < (latticeDepth / 2 + fluidDepth / 2) + 1; k++)
	//for (int j = (latticeHeight / 2 - fluidHeight / 2) - 1; j < (latticeHeight / 2 + fluidHeight / 2) + 1; j++)
	//for (int i = (latticeWidth / 2 - fluidWidth / 2) - 1; i < (latticeWidth / 2 + fluidWidth / 2) + 1; i++)
	//{
	//	if ((k == (latticeDepth / 2.0 - fluidDepth / 2) - 1 || k == (latticeDepth / 2.0 + fluidDepth / 2)) ||
	//		(j == (latticeHeight / 2.0 - fluidHeight / 2) - 1 || j == (latticeHeight / 2.0 + fluidHeight / 2)) ||
	//		(i == (latticeWidth / 2.0 - fluidWidth / 2) - 1 || i == (latticeWidth / 2.0 + fluidWidth / 2)))

	//	{
	//		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//		lattice->latticeElements[i0].cellType = lattice->latticeElements[i0].cellTypeTemp = cell_types::interphase;
	//	}
	//}

	for (int i = 0; i < lattice->getNumElements(); i++)
	{
		if ((lattice->latticeElements[i].cellType & cell_types::fluid) == cell_types::fluid ||
			(lattice->latticeElements[i].cellType & cell_types::interphase) == cell_types::interphase)
		{
			lattice->latticeElements[i].calculateInEquilibriumFunction(vectorIn, roIn);
			int test = 0;
		}
	}	

	for (int k = 0; k < latticeDepth; k++)
	for (int j = 0; j < latticeHeight; j++)
	for (int i = 0; i < latticeWidth; i++)
	if (k == 0 || k == (latticeDepth -1) || i == 0 || i == latticeWidth -1 || j == 0 || j == latticeHeight -1)
	{
		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
		lattice->latticeElements[i0].isSolid = true;
	}
			
	return 0;
}

int main( int argc, const char **argv ) {

	srand((unsigned int)time(NULL));
	// OpenGL configuration and GLUT calls  initialization
    // these need to be made before the other OpenGL
    // calls, else we get a seg fault
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (500, 500); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("1048576 points");
	glutReshapeFunc (reshape);
	glutKeyboardFunc (keys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	initGL ();
	
	if(init()==-1) 
	{
		cout << "Error opening color file\n";
		return 0;
	}

	glutDisplayFunc(display); 
	glutIdleFunc (idle);
    glutMainLoop();
	if (logFile.is_open())
		logFile.close();
	return 0;   /* ANSI C requires main to return int. */
}
