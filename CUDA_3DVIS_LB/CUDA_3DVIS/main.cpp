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

// size of the window (1024x1024)
#define DIM							512
#define BUFFER_OFFSET( i )			((char *)NULL + ( i ))
#define LOCATION_OFFSET				BUFFER_OFFSET(  0 )
#define COLOR_OFFSET				BUFFER_OFFSET( 16 )

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
int latticeWidth = 25, latticeHeight = 25, latticeDepth = 25, ncol;
float latticeTau = 0.7;
latticed3q19 *lattice;

// Entry point for CUDA Kernel execution
extern "C" void runCuda(cudaGraphicsResource** resource, Vertex* devPtr, int dim, float dt);
extern "C" void unregRes(cudaGraphicsResource** res);
extern "C" void chooseDev(int ARGC, const char **ARGV);
extern "C" void regBuffer(cudaGraphicsResource** res, unsigned int& vbo);

void display (void)
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

// set view matrix
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	glColor3f(0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, sizeof(Vertex), LOCATION_OFFSET);

	glColorPointer(4, GL_FLOAT, sizeof(Vertex), COLOR_OFFSET);

    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	    glDrawArrays(GL_POINTS, 0, DIM * DIM);
	glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glutSwapBuffers();
}

void idle(void)
{
	dt += 0.01f;
	//runCuda(&resource1, devPtr, DIM, dt);
	lattice->step();
	
	float plot_rgba, minvar=0.0, maxvar=0.2;

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	Vertex *vert_data = (Vertex*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for(int i = 0; i< lattice->getNumElements(); i++)
	{
		vert_data[i].color.x = lattice->latticeElements[i].velocityVector.x < minvar ? minvar : lattice->latticeElements[i].velocityVector.x > maxvar ? maxvar : lattice->latticeElements[i].velocityVector.x;
		vert_data[i].color.y = lattice->latticeElements[i].velocityVector.y < minvar ? minvar : lattice->latticeElements[i].velocityVector.y > maxvar ? maxvar : lattice->latticeElements[i].velocityVector.y;
		vert_data[i].color.z = 1;// lattice->latticeElements[i].velocityVector.z < 0.5 ? 0.5 : lattice->latticeElements[i].velocityVector.z > 1 ? 1 : lattice->latticeElements[i].velocityVector.z;
		vert_data[i].color.w = 1.0;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
//	delete[] vert_data;

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
	}
}

// Setting up the GL viewport and coordinate system 
void reshape (int w, int h)
{
	glViewport (0,0, w,h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	
	gluPerspective (45, w*1.0/h*1.0, 0.01, 10);
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
	FILE *fp_col;
	float rcol,gcol,bcol;
	//
    // Read in colourmap data for OpenGL display 
    //
    fp_col = fopen("cmap.dat","r");
    if (fp_col==NULL) {
	printf("Error: can't open cmap.dat \n");
	return -1;
    }
    // allocate memory for colourmap (stored as a linear array of int's)
    fscanf (fp_col, "%d", &ncol);
    cmap_rgba = (unsigned int *)malloc(ncol*sizeof(unsigned int));
    // read colourmap and store as int's
    for (int i=0;i<ncol;i++){
	fscanf(fp_col, "%f%f%f", &rcol, &gcol, &bcol);
	cmap_rgba[i]=((int)(255.0f) << 24) | // convert colourmap to int
	    ((int)(bcol * 255.0f) << 16) |
	    ((int)(gcol * 255.0f) <<  8) |
	    ((int)(rcol * 255.0f) <<  0);
    }
    fclose(fp_col);

	lattice = new latticed3q19(latticeWidth, latticeHeight, latticeDepth, latticeTau);

	for(int i =0; i< lattice->getNumElements(); i++)
		lattice->latticeElements[i].calculateInEquilibriumFunction(vector3d(1.4,1.1,1.4), 1.5);
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
