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

using namespace std;

// size of the window (1024x1024)
#define     DIM    512
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
	runCuda(&resource1, devPtr, DIM, dt);
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
    chooseDev( ARGC, ARGV );
	//creating a vertex buffer object in OpenGL and storing the handle in our global
	//variable GLuint vbo
    glGenBuffers( 1, &vbo );
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
	Vertex *vert_data = new Vertex[DIM*DIM*4];
	for( int i = 0; i < DIM*DIM; i++ )
	{
		int index = i*4;
		float x = (float)(rand()%DIM);
		float z = (float)(rand()%DIM);
		int r = rand()%2;
		int s = 1;
		if( r == 1 ) s = -1;
		vert_data[i].pos.x = (float)s * x / (float)DIM;
		s = 1;
		r = rand()%2;
		if( r == 1 ) s = -1;
		vert_data[i].pos.y = (float)s * z / (float)DIM;
		vert_data[i].pos.z = 0.0f;
		vert_data[i].pos.w = 1.0f;
		s = 1;
		r = rand()%2;
		if( r == 1 ) s = -1;
		vert_data[i].dir_speed.x = (float)s;	// Direction in X
		s = 1;
		r = rand()%2;
		if( r == 1 ) s = -1;
		vert_data[i].dir_speed.y = (float)s;	// Direction in Z
		float speed = (float)(rand()%100)+1.0f;
		speed /= 100.0f;
		vert_data[i].dir_speed.z = speed;		// Speed factor
		vert_data[i].dir_speed.w = 0.0f;

		float cr = (float)(rand()%(DIM-10))+10.0f;
		float cg = (float)(rand()%(DIM-10))+10.0f;
		float cb = (float)(rand()%(DIM-10))+10.0f;
		vert_data[i].color.x = cr / (float)DIM;
		vert_data[i].color.y = cg / (float)DIM;
		vert_data[i].color.z = cb / (float)DIM;
		vert_data[i].color.w = 1.0f;
	}
	glBufferData( GL_ARRAY_BUFFER, DIM * DIM * 4 * sizeof(Vertex), vert_data, GL_DYNAMIC_DRAW );
	delete [] vert_data;
	regBuffer(&resource1, vbo);
	runCuda(&resource1, devPtr, DIM, dt);
}


int main( int argc, const char **argv ) {

	srand((unsigned int)time(NULL));
	// OpenGL configuration and GLUT calls  initialization
    // these need to be made before the other OpenGL
    // calls, else we get a seg fault
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (DIM, DIM); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("1048576 points");
	glutReshapeFunc (reshape);
	glutKeyboardFunc (keys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	initGL ();
	initCUDA (argc, argv);
	glutDisplayFunc(display); 
	glutIdleFunc (idle);
    glutMainLoop();
	return 0;   /* ANSI C requires main to return int. */
}
