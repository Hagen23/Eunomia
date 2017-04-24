#pragma region Includes

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
// #include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include "Lattice_3D.h"
#include "Utilities_3d.h"

#pragma endregion

//extern "C" {
//	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
//}

using namespace std;

#define LATTICE_DIM					32

int type0[SIZE_3D_Z][SIZE_3D_Y][SIZE_3D_X] = { 0 };
int type1[SIZE_3D_Z][SIZE_3D_Y][SIZE_3D_X] = { 0 };
int type2[SIZE_3D_Z][SIZE_3D_Y][SIZE_3D_X] = { 0 };
int type3[SIZE_3D_Z][SIZE_3D_Y][SIZE_3D_X] = { 0 };

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

unsigned int latticeWidth = LATTICE_DIM, latticeHeight = LATTICE_DIM, latticeDepth = LATTICE_DIM, ncol;

// The lower the value, the less viscous it is. Towards 0.001 viscosity tends to 2, which in turn makes it unstable.
// The higher it is, the lower the viscosity, but mass loss is higher....
float latticeViscosity = 0.1f; 

bool withSolid = false, keypressed = false, showInterfase = true, showFluid = true, simulate = false;

//float3 vectorIn{ 0.0f, 0.0f, 0.0f};
//latticed3q19 *lattice; 

d3q19_lattice *lattice_3d;

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

	float x, y, z, posX, posY, posZ, vx, vy, vz, normMag;

// set view matrix
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
	
	glPushMatrix();
		glLineWidth(10.0);
		glBegin(GL_LINES);
		glColor3f(0, 0, 1);
		glVertex3f(0, 0, 0);
		glVertex3f(1, 0, 0);

		glColor3f(1, 0, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 1, 0);

		glColor3f(0, 1, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 0, 1);
		glEnd();
	glPopMatrix();

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
		//for(unsigned int k = 0; k  < latticeDepth; k++)
	//unsigned int k = SIZE_3D_X / 2;

	for (unsigned int slice = 0; slice < latticeDepth; slice++)
	for (unsigned int row = 0; row < latticeHeight; row++)
		for (unsigned int col = 0; col< latticeWidth; col++)
		{
			//i0 = I3D(latticeWidth, latticeHeight, i, j, k);

			posX = col / (float)latticeWidth; 
			posY = row / (float)latticeHeight; 
			posZ = slice / (float)latticeDepth;
			
			d3q19_cell *current_cell = lattice_3d->getCellAt(col, row, slice); // latticeHeight - 1 - row);

			if ((current_cell->type & CT_OBSTACLE) != cellType::CT_OBSTACLE && (current_cell->type & CT_EMPTY) != cellType::CT_EMPTY)
			{
				if ((current_cell->type & CT_FLUID) == cellType::CT_FLUID)
			//if (lattice->cell_type[i0] != cell_types::solid && lattice->cell_type[i0] != cell_types::gas)
			//{
			//	if (lattice->cell_type[i0] == cell_types::fluid)
				{
					//x = getValueFromRelation(lattice->velocityVector[i0].x);
					//y = getValueFromRelation(lattice->velocityVector[i0].y);
					//z = getValueFromRelation(lattice->velocityVector[i0].z);

					//vx = lattice->velocityVector[i0].x;
					//vy = lattice->velocityVector[i0].y;
					//vz = lattice->velocityVector[i0].z;

					x = getValueFromRelation(current_cell->velocity.x);
					y = getValueFromRelation(current_cell->velocity.y);
					z = getValueFromRelation(current_cell->velocity.z);

					vx = current_cell->velocity.x;
					vy = current_cell->velocity.y;
					vz = current_cell->velocity.z;

					glColor3f(x, y, z);
					normMag = sqrtf(vx*vx + vy*vy + vz*vz) / lattice_3d->cellSize;

					glBegin(GL_LINES);
					glVertex3f(posX, posY, posZ);
					//glVertex3f(posX + lattice->velocityVector[i0].x / normMag, posY + lattice->velocityVector[i0].y / normMag,
					//	posZ + lattice->velocityVector[i0].z / normMag);
					glVertex3f(
						posX + current_cell->velocity.x / normMag,
						posY + current_cell->velocity.y / normMag,
						posZ + current_cell->velocity.z / normMag);
					glEnd();
				}
				else
				{
					if (showInterfase)
					//if (lattice->cell_type[i0] & cell_types::interfase)
					if ((current_cell->type & CT_INTERFACE) == cellType::CT_INTERFACE)
					{
						glPointSize(2.0);
						glColor3f(1, 1, 0);
						glBegin(GL_POINTS);
						glVertex3f(posX, posY, posZ);
						glEnd();
					}
				}
			}

			//if ((current_cell->type & CT_OBSTACLE) == cellType::CT_OBSTACLE)
			//{
			//	glPointSize(2.0);
			//	glColor3f(0, 1, 1);
			//	glBegin(GL_POINTS);
			//	glVertex3f(posX, posY, posZ);
			//	glEnd();
			//}
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
		lattice_3d->step();
		//lattice->step();

	//if(keypressed)
	//{
	//	if(withSolid)
	//	{
	//			// Solid inside the cube
	//		for(int k = latticeDepth/4; k < latticeDepth/2.0 + latticeDepth/4; k++)
	//			for(int j = latticeHeight/4; j< latticeHeight/2.0 +latticeHeight/4; j++)
	//				for(int i = latticeWidth/4; i< latticeWidth/2.0 + latticeWidth/4; i++)
	//				{
	//					int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//					lattice->solid[i0] = 1;
	//				}

	//		keypressed = false;
	//	}
	//	else 
	//	{
	//			// Solid inside the cube
	//		for(int k = latticeDepth/4; k < latticeDepth/2.0 + latticeDepth/4; k++)
	//			for(int j = latticeHeight/4; j< latticeHeight/2.0 +latticeHeight/4; j++)
	//				for(int i = latticeWidth/4; i< latticeWidth/2.0 + latticeWidth/4; i++)
	//				{
	//					int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//					lattice->solid[i0] = 0;
	//				}

	//		keypressed = false;
	//	}
	//}
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

			printf("\nMin mass exchange %f\n", lattice_3d->minMassExchange);
			printf("\nCounter %f\n", lattice_3d->exchange_counter);

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
			cout << "Streaming: " << simulate << endl;
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
void initTypes(int *types)
{
	for (int slice = 0; slice < SIZE_3D_Z; slice++)
	{
		for (int row = 0; row < SIZE_3D_Y; row++)
		{
			for (int col = 0; col < SIZE_3D_X; col++)
			{
				//if (((slice <= 19 && slice >= 11) && (row <= 26 && row >= 18) && (col <= 19 && col >= 11)))
				//	types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_FLUID;
				//else if (
				//	(
				//	((slice == 20 || slice == 10) && (row <= 27 && row >= 17) && (col <= 20 && col >= 10)) ||
				//	((slice <= 20 && slice >= 10) && (row == 27 || row == 17) && (col <= 20 && col >= 10)) ||
				//	((slice <= 20 && slice >= 10) && (row <= 27 && row >= 17) && (col == 20 || col == 10))
				//	)
				//	)
				//	types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_INTERFACE;
				//else
				//	types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_EMPTY;

				if (slice <= 5)
					types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_FLUID;
				else if (slice == 6)
					types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_INTERFACE;
				else
					types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_EMPTY;

				if ((slice == 0 || slice == SIZE_3D_Z - 1) || (row == 0 || row == SIZE_3D_Y - 1) || (col == 0 || col == SIZE_3D_X - 1))
					types[I3D(SIZE_3D_X, SIZE_3D_Y, col, row, slice)] = cellType::CT_OBSTACLE;
			}
		}
	}
}

// glew (extension loading), OpenGL state and CUDA - OpenGL interoperability initialization
void initGL ()
{
	/*GLenum err = glewInit();
	if (GLEW_OK != err) 
		return;
	cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << endl;*/
	const GLubyte* renderer;
	const GLubyte* version;
	//const GLubyte* glslVersion;

	renderer = glGetString(GL_RENDERER); /* get renderer string */
	version = glGetString(GL_VERSION); /* version as a string */
	//glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
	//printf("GLSL version supported %s\n", glslVersion);

	glEnable(GL_DEPTH_TEST);
}

int init(void)
{
	int dimension = 16;
	int fluidWidth = dimension, fluidHeight = dimension, fluidDepth = dimension;
	
	int *types = new int[SIZE_3D_Z * SIZE_3D_Y * SIZE_3D_X]();
	//types = new int**[SIZE_3D_Z];
	//for (int slice = 0; slice < SIZE_3D_Z; slice++)
	//{
	//	types[slice] = new int *[SIZE_3D_Y];
	//	for (int row = 0; row < SIZE_3D_Y; row++)
	//		types[slice][row] = new int[SIZE_3D_X]();
	//}
	
	initTypes(types);

	//for (int slice = 0; slice < SIZE_3D_Z; slice++)
	//for (int row = 0; row < SIZE_3D_Y; row++)
	//for (int col = 0; col < SIZE_3D_X; col++)
	//	types[slice][SIZE_3D_Y - row - 1][col] = type0[slice][row][col];

	//lattice = new latticed3q19(latticeWidth, latticeHeight, latticeDepth, latticeViscosity, LATTICE_DIM, 1.0f);

	lattice_3d = new d3q19_lattice(SIZE_3D_X, SIZE_3D_Y, SIZE_3D_Z, latticeViscosity, SIZE_3D_X, 1.0f);

	lattice_3d->initCells(types, 1.f, float3{ 0.f,0.f,0.f });

	//lattice_3d->print_types();

	lattice_3d->print_fluid_amount();
	
	//for (unsigned int k = latticeDepth / 2 - fluidDepth / 2; k < (latticeDepth / 2 + fluidDepth / 2); k++)
	//for (unsigned int j = latticeHeight / 2 - fluidHeight / 2; j< (latticeHeight / 2 + fluidHeight / 2); j++)
	//for (unsigned int i = latticeWidth / 2 - fluidWidth / 2; i< (latticeWidth / 2 + fluidWidth / 2); i++)
	//{
	//	int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//	lattice->cell_type[i0] = lattice->cell_type_temp[i0] = (cell_types::fluid);
	//}

	//for (unsigned int i = 0; i < latticeWidth; i++)
	//for (unsigned int j = 0; j < latticeHeight; j++)
	//for (unsigned int k = 0; k < latticeDepth; k++)
	//{
	//	int cellIndex = I3D(latticeWidth, latticeHeight, i, j, k);
	//	if (lattice->cell_type[cellIndex] == cell_types::fluid)
	//	{
	//		for (int l = 0; l < 19; l++)
	//		{
	//			int nbCellIndex = I3D(latticeWidth, latticeHeight, 
	//				(int)(i + speedDirection[l].x), (int)(j + speedDirection[l].y),	(int)(k + speedDirection[l].z));
	//			if (lattice->cell_type[nbCellIndex] == cell_types::gas)
	//				lattice->cell_type[nbCellIndex] = lattice->cell_type_temp[nbCellIndex] = cell_types::interfase;
	//		}
	//	}
	//}
	//for (unsigned int k = (latticeDepth / 2 - fluidDepth / 2) - 1; k < (latticeDepth / 2 + fluidDepth / 2) + 1; k++)
	//for (unsigned int j = (latticeHeight / 2 - fluidHeight / 2) - 1; j < (latticeHeight / 2 + fluidHeight / 2) + 1; j++)
	//for (unsigned int i = (latticeWidth / 2 - fluidWidth / 2) - 1; i < (latticeWidth / 2 + fluidWidth / 2) + 1; i++)
	//{
	//	if ((k == (latticeDepth / 2.0 - fluidDepth / 2) - 1 || k == (latticeDepth / 2.0 + fluidDepth / 2)) ||
	//		(j == (latticeHeight / 2.0 - fluidHeight / 2) - 1 || j == (latticeHeight / 2.0 + fluidHeight / 2))||
	//		(i == (latticeWidth / 2.0 - fluidWidth / 2) - 1 || i == (latticeWidth / 2.0 + fluidWidth / 2)))
	//		
	//	{
	//		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//		lattice->cell_type[i0] = lattice->cell_type_temp[i0] = cell_types::interfase;
	//	}
	//}

	//for (int i = 0; i < lattice->getNumElements(); i++)
	//lattice->initLatticeDistributions();

    //lattice->calculateInitialMass();

	//for (unsigned int k = 0; k < latticeDepth; k++)
	//for (unsigned int j = 0; j < latticeHeight; j++)
	//for (unsigned int i = 0; i < latticeWidth; i++)
	//{
	//	if (k == 0 || k == (latticeDepth - 1) || i == 0 || i == latticeWidth - 1 || j == 0 || j == latticeHeight - 1)
	//	{
	//		int i0 = I3D(latticeWidth, latticeHeight, i, j, k);
	//		lattice->cell_type[i0] = cell_types::solid;
	//	}
	//}

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
	init();

	glutDisplayFunc(display); 
	glutIdleFunc (idle);
    glutMainLoop();

	return 0;   /* ANSI C requires main to return int. */
}
