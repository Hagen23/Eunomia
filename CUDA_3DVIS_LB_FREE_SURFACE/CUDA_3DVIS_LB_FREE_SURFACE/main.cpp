#pragma once

#pragma region Includes

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

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

#include "lb_src/Lattice.h"

#pragma endregion

//extern "C" {
//	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
//}

using namespace std;

#define LATTICE_DIM					32
#define LATTICE_MASS				(float)50 // Units?

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

unsigned int latticeWidth = LATTICE_DIM, latticeHeight = LATTICE_DIM, latticeDepth = LATTICE_DIM, ncol;
float latticeViscosity = 15.0f, roIn = (float)LATTICE_DIM / (float)LATTICE_MASS; // 0.1f;
bool withSolid = false, keypressed = false, showInterfase = true, showFluid = true, simulate = false;

float3 vectorIn{ 0, 0, 0 };
latticed3q19 *lattice; 

float cubeFaces[24][3] = 
{
	{0.0,0.0,0.0}, {1.0,0.0,0.0}, {1.0,1.0,0.0}, {0.0,1.0,0.0},
	{0.0,0.0,0.0}, {1.0,0.0,0.0}, {1.0,0,1.0}, {0.0,0,1.0},
	{0.0,0.0,0.0}, {0,1.0,0.0}, {0,1.0,1.0}, {0.0,0,1.0},
	{0.0,1.0,0.0}, {1.0,1.0,0.0}, {1.0,1.0,1.0}, {0.0,1.0,1.0},
	{1.0,0.0,0.0}, {1.0,1.0,0.0}, {1.0,1.0,1.0}, {1.0,0.0,1.0},
	{0.0,0.0,1.0}, {0,1.0,1.0}, {1.0,1.0,1.0}, {1.0,0,1.0}
};

float getValueFromRelation(float value, float minColorVar=0.0f, float maxColorVar=1.0, float minVelVar = -0.01, float maxVelVar = 0.01);

#pragma region Display and scene control
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
		for(unsigned int k = 0; k  < latticeDepth; k++)
		for (unsigned int j = 0; j < latticeHeight; j++)
		for (unsigned int i = 0; i < latticeWidth; i++)
		{
			i0 = I3D(latticeWidth, latticeHeight, i, j, k);

			posX = i / (float)latticeWidth; posY =  j / (float)latticeHeight; posZ = k / (float)latticeDepth;

			if (!lattice->solid[i0] && (lattice->cellType[i0] & cell_types::fluid) && showFluid)
			{
				x = getValueFromRelation(lattice->velocityVector[i0].x);
				y = getValueFromRelation(lattice->velocityVector[i0].y);
				z = getValueFromRelation(lattice->velocityVector[i0].z);

				vx = lattice->velocityVector[i0].x;
				vy = lattice->velocityVector[i0].y;
				vz = lattice->velocityVector[i0].z;

				glColor3f(x,y,z);
				normMag = sqrtf(vx*vx + vy*vy + vz*vz)*10; 

				glBegin(GL_LINES);
					glVertex3f(posX, posY, posZ);
					glVertex3f(posX + lattice->velocityVector[i0].x / normMag, posY + lattice->velocityVector[i0].y / normMag,
						posZ + lattice->velocityVector[i0].z / normMag);
				glEnd();
			}
			else
			{
				if (showInterfase)
				if (lattice->cellType[i0] & cell_types::interphase)
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
		lattice->step();

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
						lattice->solid[i0] = 1;
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
						lattice->solid[i0] = 0;
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
            exit(0);
			break;
		case 'a':
			keypressed = true;
			withSolid = !withSolid;
			break;
		case 'i':
			showInterfase = !showInterfase;
			break;
		case 'f':
			showFluid = !showFluid;
			break;
		case 32:
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

#pragma endregion

#pragma region Initialization
// glew (extension loading), OpenGL state and CUDA - OpenGL interoperability initialization
void initGL ()
{
	/*GLenum err = glewInit();
	if (GLEW_OK != err) 
		return;
	cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << endl;*/
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
	int dimension = 16;
	int fluidWidth = dimension, fluidHeight = dimension, fluidDepth = dimension;

	roIn = LATTICE_MASS / (dimension * dimension * dimension);

	lattice = new latticed3q19(latticeWidth, latticeHeight, latticeDepth, latticeViscosity, LATTICE_MASS, LATTICE_DIM, 1.0f);
	
	for (unsigned int k = latticeDepth / 2 - fluidDepth / 2; k < latticeDepth / 2 + fluidDepth / 2; k++)
	for (unsigned int j = latticeHeight / 2 - fluidHeight / 2; j< latticeHeight / 2 + fluidHeight / 2; j++)
	for (unsigned int i = latticeWidth / 2 - fluidWidth / 2; i< latticeWidth / 2 + fluidWidth / 2; i++)
	{
		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
		lattice->cellType[i0] = lattice->cellTypeTemp[i0] = (cell_types::fluid);
	}

	for (unsigned int k = (latticeDepth / 2 - fluidDepth / 2) - 1; k < (latticeDepth / 2 + fluidDepth / 2) + 1; k++)
	for (unsigned int j = (latticeHeight / 2 - fluidHeight / 2) - 1; j < (latticeHeight / 2 + fluidHeight / 2) + 1; j++)
	for (unsigned int i = (latticeWidth / 2 - fluidWidth / 2) - 1; i < (latticeWidth / 2 + fluidWidth / 2) + 1; i++)
	{
		if ((k == (latticeDepth / 2.0 - fluidDepth / 2) - 1 || k == (latticeDepth / 2.0 + fluidDepth / 2)) ||
			(j == (latticeHeight / 2.0 - fluidHeight / 2) - 1 || j == (latticeHeight / 2.0 + fluidHeight / 2))||
			(i == (latticeWidth / 2.0 - fluidWidth / 2) - 1 || i == (latticeWidth / 2.0 + fluidWidth / 2)))
			
		{
			int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
			lattice->cellType[i0] = lattice->cellTypeTemp[i0] = cell_types::interphase;
		}
	}

	for (int i = 0; i < lattice->getNumElements(); i++)
		lattice->calculateInEquilibriumFunction(i, vectorIn, roIn);

    //lattice->calculateInitialMass();

	for (unsigned int k = 0; k < latticeDepth; k++)
	for (unsigned int j = 0; j < latticeHeight; j++)
	for (unsigned int i = 0; i < latticeWidth; i++)
	{
		if (k == 0 || k == (latticeDepth - 1) || i == 0 || i == latticeWidth - 1 || j == 0 || j == latticeHeight - 1)
		{
			int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
			lattice->solid[i0] = 1;
		}
	}

	return 0;
}

#pragma endregion

int main( int argc, const char **argv ) {

	srand((unsigned int)time(NULL));
	// OpenGL configuration and GLUT calls  initialization
    // these need to be made before the other OpenGL
    // calls, else we get a seg fault
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (500, 500); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("CUDA 3DVIS LB");
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
	return 0;   /* ANSI C requires main to return int. */
}
