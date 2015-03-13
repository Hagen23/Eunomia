////////////////////////////////////////////////////////////////////////////////
// Crude 2D Lattice Boltzmann Demo program
// C version
// Graham Pullan - Oct 2008
//
//      f6  f2   f5
//        \  |  /
//         \ | /
//          \|/
//      f3---|--- f1
//          /|\
//         / | \       and f0 for the rest (zero) velocity
//        /  |  \
//      f7  f4   f8
//
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../GL/glew.h"
#include "../GL/glut.h"
#include "lb_src/Lattice.h"

//Function to linearly go over the arrays


////////////////////////////////////////////////////////////////////////////////

// OpenGL pixel buffer object and texture //
GLuint gl_PBO, gl_Tex;


// arrays //

latticed2q9 *lattice;

float *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8; //9 arrays for each of the possible directions, each array will hold ni*nj values
float *tmpf0,*tmpf1,*tmpf2,*tmpf3,*tmpf4,*tmpf5,*tmpf6,*tmpf7,*tmpf8;
float *cmap,*plotvar;
int *solid;
unsigned int *cmap_rgba, *plot_rgba;  //rgba arrays for plotting

bool withSolid = false, keypressed = false;

bool in_bc = false;
bool leftDown = false;

// scalars //
float tau,faceq1,faceq2,faceq3; 
float vxin, vyin, vzin, roout;
float width, height;
int latticeWidth, latticeHeight;
int ni,nj;
int ncol;
int ipos_old,jpos_old, draw_solid_flag;

////////////////////////////////////////////////////////////////////////////////

//
// OpenGL function prototypes 
//
void display(void);
void resize(int w, int h);
void mouse(int button, int state, int x, int y);
void mouse_motion(int x, int y);
void keyboard (unsigned char key, int x, int y);
void shutdown(void);

//
// Lattice Boltzmann function prototypes
//
void stream(void);
void collide(void);
void solid_BC(void);
void per_BC(void);
void in_BC(void);
void ex_BC_crude(void);
void apply_BCs(void);

unsigned int get_col(float min, float max, float val);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initLattice()
{
	vector3d vectorIn(vxin, vyin, vzin);

	lattice = new latticed2q9(latticeWidth, latticeHeight, 1, tau);

	for(int j = nj/4; j < nj/2 + nj/4; j++)
		for(int i = ni /4; i < ni/2 +  ni/4; i++)
		{
			int i0 = I2D(ni, i, j);
			lattice->latticeElements[i0].isSolid = true;
		}

	for(int i =0; i < latticeWidth *  latticeHeight; i++)
	{
		lattice->latticeElements[i].calculateInEquilibriumFunction(vectorIn, roout);
		plotvar[i] = (vxin + vyin);
		//plotvar[i] = sqrt(vectorIn.dotProduct(vectorIn));
	}
}

int main(int argc, char **argv)
{
    int array_size_2d,totpoints,i;
    float rcol,gcol,bcol;
	    
	FILE *fp_col;

    // The following parameters are usually read from a file, but
    // hard code them for the demo:
	ni = latticeWidth =200; //Width
    nj = latticeHeight = 200; //Height
    vxin = 0.05;
	vyin = 0.05;
	vzin = 0.0;
    roout=1.50;
    tau=1.50;
    // End of parameter list

    // Write parameters to screen
    printf ("ni = %d\n", ni);
    printf ("nj = %d\n", nj);
    printf ("vxin = %f\n", vxin);
    printf ("roout = %f\n", roout);
    printf ("tau = %f\n", tau);
    
	totpoints=ni*nj;    
    array_size_2d=ni*nj*sizeof(float);

    // Allocate memory for arrays
    
    plotvar = (float*)malloc(array_size_2d);
    
    plot_rgba = (unsigned int*)malloc(ni*nj*sizeof(unsigned int));

	initLattice();
	 //
    // Read in colourmap data for OpenGL display 
    //
    fp_col = fopen("cmap.dat","r");
    if (fp_col==NULL) {
	printf("Error: can't open cmap.dat \n");
	return 1;
    }
    // allocate memory for colourmap (stored as a linear array of int's)
    fscanf (fp_col, "%d", &ncol);
    cmap_rgba = (unsigned int *)malloc(ncol*sizeof(unsigned int));
    // read colourmap and store as int's
    for (i=0;i<ncol;i++){
	fscanf(fp_col, "%f%f%f", &rcol, &gcol, &bcol);
	cmap_rgba[i]=((int)(255.0f) << 24) | // convert colourmap to int
	    ((int)(bcol * 255.0f) << 16) |
	    ((int)(gcol * 255.0f) <<  8) |
	    ((int)(rcol * 255.0f) <<  0);
    }
    fclose(fp_col);


    //
    // Iinitialise OpenGL display - use glut
    //
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(ni, nj);      // Window of ni x nj pixels
    glutInitWindowPosition(50, 50);  // position
    glutCreateWindow("2D LB");       // title

    // Check for OpenGL extension support 
    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
    if(!glewIsSupported(
                        "GL_VERSION_2_0 " 
                        "GL_ARB_pixel_buffer_object "
                        "GL_EXT_framebuffer_object "
                        )){
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return 0;
    }

    // Set up view
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,ni,0.,nj, -200.0, 200.0);


    // Create texture which we use to display the result and bind to gl_Tex
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);                     // Generate 2D texture
    glBindTexture(GL_TEXTURE_2D, gl_Tex);          // bind to gl_Tex
    // texture properties:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ni, nj, 0, 
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);


    // Create pixel buffer object and bind to gl_PBO. We store the data we want to
    // plot in memory on the graphics card - in a "pixel buffer". We can then 
    // copy this to the texture defined above and send it to the screen
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    printf("Buffer created.\n");
    

    // Set the call-back functions and start the glut loop
    printf("Starting GLUT main loop...\n");
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutIdleFunc(display);
    glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
    glutMotionFunc(mouse_motion); 
    glutMainLoop();

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void stream(void)

// Move the f values one grid spacing in the directions that they are pointing
// i.e. f1 is copied one location to the right, etc.

{
	lattice->stream();
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void collide(void)

// Collisions between the particles are modeled here. We use the very simplest
// model which assumes the f's change toward the local equlibrium value (based
// on density and velocity at that point) over a fixed timescale, tau	 

{
	lattice->collide();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void per_BC(void)

// All the f's leaving the bottom of the domain (j=0) enter at the top (j=nj-1),
// and vice-verse

{
    int i0,i1,i;

    for (i=0; i<ni; i++){
	i0 = I2D(ni,i,0);
	i1 = I2D(ni,i,nj-1);
	f2[i0] = f2[i1];
	f5[i0] = f5[i1];
	f6[i0] = f6[i1];
	f4[i1] = f4[i0];
	f7[i1] = f7[i0];
	f8[i1] = f8[i0];
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void solid_BC(void)

// This is the boundary condition for a solid node. All the f's are reversed -
// this is known as "bounce-back"

{
    int i0 = 0;

	for(int j = 0; j < latticeHeight; j++)
	{
		for(int i = 0; i<latticeWidth; i++)
		{
			i0 = I2D(latticeWidth, i, j);
			
			if(!lattice->latticeElements[i0].isSolid)
			{
				lattice->solid_BC(i0);
			}
			/*else
				solid_BC(i0);*/
		}
	}
	
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void in_BC(void)

// This inlet BC is extremely crude but is very stable
// We set the incoming f values to the equilibirum values assuming:
// ro=roout; vx=vxin; vy=0

{
	if(in_bc)
	{
		int i0, j, i;
		float f1new, f5new, f8new, f7new, f4new, vx_term;

		vx_term = 1.f + 3.f*(vxin + vyin) +3.f*(vxin + vyin)*(vxin + vyin)  - 1.5 * ( vxin + vyin)  * ( vxin + vyin);
		//vx_term = 1.f + 3.f*vxin +3.f*vxin*vxin;
	
		f1new = roout * faceq2 * vx_term;
		f5new = roout * faceq3 * vx_term;
		f8new = f5new;

		/*f1new = roout * faceq2 * (1.f + 3.f * (vxin) + 4.5f * vxin * vxin - 1.5 * ( vxin + vxin)  * ( vxin + vxin));
	    f5new = roout * faceq3 * (1.f + 3.f * ( vxin + vyin) + 4.5f * ( vxin +vyin) * ( vxin + vyin) - 1.5 * ( vxin + vyin)  * ( vxin + vyin));
	    f8new = roout * faceq3 * (1.f + 3.f * (vxin - vyin) + 4.5f *  (vxin - vyin) *  (vxin - vyin) -  1.5 * ( vxin + vyin)  * ( vxin + vyin));*/

		f4new = roout * faceq2 * (1.f + 3.f * vyin + 4.5f * vyin * vyin - 1.5 * ( vxin + vyin)  * ( vxin + vyin));
	    f7new = roout * faceq3 * (1.f + 3.f * ( - vxin - vyin) + 4.5f * ( - vxin - vyin) * ( - vxin - vyin) - 1.5 * ( vxin + vyin)  * ( vxin + vyin));
	    f8new = roout * faceq3 * (1.f + 3.f * (vxin - vyin) + 4.5f *  (vxin - vyin) *  (vxin - vyin) -  1.5 * ( vxin + vyin)  * ( vxin + vyin));
		
		/*for (j=0; j<nj; j++){
		  i0 = I2D(ni,0,j);
		  f1[i0] = f1new;
		  f5[i0] = f5new;
		  f8[i0] = f8new;
		}*/

		for (j=0; j<nj; j++)
		//for (i=0; i<ni; i++)
		{
		  i0 = I2D(ni,0,j);
		  f1[i0] = f1new;
		  f5[i0] = f5new;
		  f8[i0] = f8new;
		}

		//for (j=0; j<nj; j++)
		for (i=0; i<ni; i++)
		{
		  i0 = I2D(ni,i,0);
		  f7[i0] = f7new;
		  f4[i0] = f4new;
		  f8[i0] = f8new;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void ex_BC_crude(void)

// This is the very simplest (and crudest) exit BC. All the f values pointing
// into the domain at the exit (ni-1) are set equal to those one node into
// the domain (ni-2)

{
    int i0, i1, j;

    for (j=0; j<nj; j++){
	i0 = I2D(ni,ni-1,j);
	i1 = i0 - 1;
	f3[i0] = f3[i1];
	f6[i0] = f6[i1];
	f7[i0] = f7[i1];
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Boundary conditions
void apply_BCs(void)

// Just calls the individual BC functions

{
    //per_BC();

    solid_BC();
	 	
    //in_BC();

   // ex_BC_crude();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void display(void)

// This function is called automatically, over and over again,  by GLUT 

{
    int i,j,ip1,jp1,i0,icol,i1,i2,i3,i4,isol;
    float minvar,maxvar,frac;

    // set upper and lower limits for plotting
    minvar=0.0;
    maxvar=0.2;

	if(keypressed)
	{
		if(withSolid)
		{
			for(int j = nj/4; j < nj/2 + nj/4; j++)
				for(int i = ni /4; i < ni/2 +  ni/4; i++)
				{
					int i0 = I2D(ni, i, j);
					lattice->latticeElements[i0].isSolid = true;
				}

			keypressed = false;
		}
		else 
		{
			for(int j = nj/4; j < nj/2 + nj/4; j++)
				for(int i = ni /4; i < ni/2 +  ni/4; i++)
				{
					int i0 = I2D(ni, i, j);
					lattice->latticeElements[i0].isSolid = false;
				}

			keypressed = false;
		}
	}

    // do one Lattice Boltzmann step: stream, BC, collide:
    stream();
    //apply_BCs();
    collide();
	
	float vx, vy;

	for (i=0;i<ni*nj;i++){
		vx = lattice->latticeElements[i].velocityVector.x;
		vy = lattice->latticeElements[i].velocityVector.y;
		plotvar[i] = sqrt(vx*vx + vy*vy);
	}
    // convert the plotvar array into an array of colors to plot
    // if the mesh point is solid, make it black
    for (j=0;j<nj;j++){
	for (i=0;i<ni;i++){
	    i0=I2D(ni,i,j);
	    frac=(plotvar[i0]-minvar)/(maxvar-minvar);
	    icol=frac*ncol;
		isol=lattice->latticeElements[i0].isSolid?0:1;
	    plot_rgba[i0] = isol*cmap_rgba[icol];   
	}
    }

    // Fill the pixel buffer with the plot_rgba array
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,ni*nj*sizeof(unsigned int),
		 (void **)plot_rgba,GL_STREAM_COPY);

    // Copy the pixel buffer to the texture, ready to display
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,ni,nj,GL_RGBA,GL_UNSIGNED_BYTE,0);

    // Render one quad to the screen and colour it using our texture
    // i.e. plot our plotvar data to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f (0.0, 0.0);
    glVertex3f (0.0, 0.0, 0.0);
    glTexCoord2f (1.0, 0.0);
    glVertex3f (ni, 0.0, 0.0);
    glTexCoord2f (1.0, 1.0);
    glVertex3f (ni, nj, 0.0);
    glTexCoord2f (0.0, 1.0);
    glVertex3f (0.0, nj, 0.0);
    glEnd();
    glutSwapBuffers();

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void resize(int w, int h)

// GLUT resize callback to allow us to change the window size

{
   width = w;
   height = h;
   glViewport (0, 0, w, h); 
   glMatrixMode (GL_PROJECTION); 
   glLoadIdentity (); 
   glOrtho (0., ni, 0., nj, -200. ,200.); 
   glMatrixMode (GL_MODELVIEW); 
   glLoadIdentity ();
}
    

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void keyboard (unsigned char key, int x, int y)
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
	}
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void mouse(int button, int state, int x, int y)

// GLUT mouse callback. Left button draws the solid, right button removes solid

{
    float xx,yy;
	
    if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
		leftDown = true;
        draw_solid_flag = 0;
        xx=x;
        yy=y;
        ipos_old=xx/width*ni;
        jpos_old=(height-yy)/height*nj;
    }

    if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN) && !leftDown) {
        /*draw_solid_flag = 1;
        xx=x;
        yy=y;
        ipos_old=xx/width*ni;
        jpos_old=(height-yy)/height*nj;*/
		in_bc = !in_bc;
		std::cout << "inbc " << in_bc << std::endl;
    }

	if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
		leftDown = false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void mouse_motion(int x, int y)

// GLUT call back for when the mouse is moving
// This sets the solid array to draw_solid_flag as set in the mouse callback
// It will draw a staircase line if we move more than one pixel since the
// last callback - that makes the coding a bit cumbersome:

{
	if(leftDown)
	{
		float xx,yy,frac;
		int ipos,jpos,i,j,i1,i2,j1,j2, jlast, jnext;
		xx=x;
		yy=y;
		ipos=(int)(xx/width*(float)ni);
		jpos=(int)((height-yy)/height*(float)nj);

		if (ipos <= ipos_old){
			i1 = ipos;
			i2 = ipos_old;
			j1 = jpos;
			j2 = jpos_old;
		}
		else {
			i1 = ipos_old;
			i2 = ipos;
			j1 = jpos_old;
			j2 = jpos;
		}
    
		jlast=j1;

		for (i=i1;i<=i2;i++){
			if (i1 != i2) {
				frac=(float)(i-i1)/(float)(i2-i1);
				jnext=(int)(frac*(j2-j1))+j1;
			}
			else {
				jnext=j2;
			}
			/*if (jnext >= jlast) {
				solid[I2D(ni,i,jlast)]=draw_solid_flag;
				for (j=jlast; j<=jnext; j++){
					solid[I2D(ni,i,j)]=draw_solid_flag;
				}
			}
			else {
				solid[I2D(ni,i,jlast)]=draw_solid_flag;
				for (j=jnext; j<=jlast; j++){
					solid[I2D(ni,i,j)]=draw_solid_flag;
				}
			}*/
			jlast = jnext;
		}

    
		ipos_old=ipos;
		jpos_old=jpos;
	}
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////




