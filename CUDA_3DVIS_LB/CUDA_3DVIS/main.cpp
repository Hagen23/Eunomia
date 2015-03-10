#pragma once

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

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include "lb_src\Lattice.h"

using namespace std;

// size of the window (102.04x102.04)
#define DIM							512
#define BUFFER_OFFSET( i )			((char *)NULL + ( i ))
#define LOCATION_OFFSET				BUFFER_OFFSET(  0 )
#define COLOR_OFFSET				BUFFER_OFFSET( 16 )
#define LATTICE_DIM					20

// global variables that will store handles to the data we
// intend to share between OpenGL and CUDA calculated data.
// handle for OpenGL side:
unsigned int vbo;	// VBO for storing positions.
// handle for CUDA side:
cudaGraphicsResource *resource1;

typedef struct
{
	float4 pos;
	float4 color;
	float4 dir_speed;
} Vertex;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

Vertex* devPtr;
float dt = 0;

unsigned int *cmap_rgba, *plot_rgba;  //rgba arrays for plotting
int latticeWidth = LATTICE_DIM, latticeHeight = LATTICE_DIM, latticeDepth = LATTICE_DIM, ncol;
double latticeTau = 4.0, roIn = 1.0;
bool withSolid = false, keypressed = false;
vector3d vectorIn(0.05, 0.01, 0.03);
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

// Entry point for CUDA Kernel execution
extern "C" void runCuda(cudaGraphicsResource** resource, Vertex* devPtr, int dim, float dt);
extern "C" void unregRes(cudaGraphicsResource** res);
extern "C" void chooseDev(int ARGC, const char **ARGV);
extern "C" void regBuffer(cudaGraphicsResource** res, unsigned int& vbo);

float getValueFromRelation(float value, float minColorVar=0.01, float maxColorVar=1.0, float minVelVar = -1.0, float maxVelVar = 1.0);

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

			if(!lattice->latticeElements[i0].isSolid)
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
			else
			{
				glColor3f(1.0, 1.0, 0.0);
				glPointSize(2.0);
				glBegin(GL_POINTS);
					glVertex3f(posX, posY, posZ);
				glEnd();
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
	//dt += 0.01f;
	////runCuda(&resource1, devPtr, DIM, dt);
	lattice->step();
	//
	//float plot_rgba, minColorVar=0.2.0, maxColorVar=0.9, minVelVar = -1000, maxVelVar = 1000;

	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//Vertex *vert_data = (Vertex*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	//for(int i = 0; i< lattice->getNumElements(); i++)
	//{		
	//	if(!(lattice->latticeElements[i].velocityVector.x == lattice->latticeElements[i].velocityVector.x ))
	//	{
	//		cout << endl;
	//	}

	//	if(lattice->latticeElements[i].isSolid)
	//	{
	//		vert_data[i].color.x = 1;
	//		vert_data[i].color.y = 1;
	//		vert_data[i].color.z = 0;//getValueFromRelation(lattice->latticeElements[i].velocityVector.z);
	//		vert_data[i].color.w = 1.0;
	//	}
	//	else
	//	{
	//		//float x = getValueFromRelation(lattice->latticeElements[i].velocityVector.x);
	//		//float y = getValueFromRelation(lattice->latticeElements[i].velocityVector.y);
	//		//float z = getValueFromRelation(lattice->latticeElements[i].velocityVector.z);
	//		
	//		vert_data[i].color.x = lattice->latticeElements[i].velocityVector.x;
	//		vert_data[i].color.y = lattice->latticeElements[i].velocityVector.y;
	//		vert_data[i].color.z = lattice->latticeElements[i].velocityVector.z;
	//		vert_data[i].color.w = 1.0;
	//	}
	//}

	//glUnmapBuffer(GL_ARRAY_BUFFER);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

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
            // clean up OpenGL and CUDA
            unregRes( &resource1 );
            glDeleteBuffers( 1, &vbo );
            exit(0);
			break;
		case 'a':
			keypressed = true;
			withSolid = !withSolid;
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
	
}

void initCUDA (int ARGC, const char **ARGV)
{
	int i0 = 0;
    chooseDev( ARGC, ARGV );
	//creating a vertex buffer object in OpenGL and storing the handle in our global
	//variable GLuint vbo
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
	Vertex *vert_data = new Vertex[lattice->getNumElements()];
	for(int k = 0; k  < latticeDepth; k++)
	for(int j = 0; j < latticeHeight; j++)
	for( int i = 0; i < latticeWidth; i++ )
	{
		i0 = I3D(latticeWidth, latticeHeight, i, j, k);

		vert_data[i0].pos.x = i / (float)latticeWidth;
		vert_data[i0].pos.y = j / (float)latticeHeight;
		vert_data[i0].pos.z = k / (float)latticeDepth;
		vert_data[i0].pos.w = 1.0f;

		float cr = (float)(rand()%(DIM-10))+10.0f;
		float cg = (float)(rand()%(DIM-10))+10.0f;
		float cb = (float)(rand()%(DIM-10))+10.0f;

		vert_data[i0].color.x = cr / (float)DIM;
		vert_data[i0].color.y = cg / (float)DIM;
		vert_data[i0].color.z = cb / (float)DIM;
		vert_data[i0].color.w = 1.0f;
	}
	glBufferData( GL_ARRAY_BUFFER, lattice->getNumElements() * sizeof(Vertex), vert_data, GL_DYNAMIC_DRAW );
	delete [] vert_data;
	//regBuffer(&resource1, vbo);
	//runCuda(&resource1, devPtr, DIM, dt);
}

int init(void)
{
	lattice = new latticed3q19(latticeWidth, latticeHeight, latticeDepth, latticeTau);

	for(int i =0; i< lattice->getNumElements(); i++)
		lattice->latticeElements[i].calculateInEquilibriumFunction(vectorIn, roIn);

	//for(int k = latticeDepth/4; k < latticeDepth/2.0 + latticeDepth/4; k++)
	//	for(int j = latticeHeight/4; j< latticeHeight/2.0 +latticeHeight/4; j++)
	//		for(int i = latticeWidth/4; i< latticeWidth/2.0 + latticeWidth/4; i++)
	//		{
	//			int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//			lattice->latticeElements[i0].isSolid = true;
	//		}

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

	initCUDA (argc, argv);
	glutDisplayFunc(display); 
	glutIdleFunc (idle);
    glutMainLoop();
	return 0;   /* ANSI C requires main to return int. */
}
