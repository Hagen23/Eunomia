#include "Lattice.h"

latticed2q9::latticed2q9(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd2q9[_numberElements];
}
	
latticed2q9::~latticed2q9()
{
	delete[] latticeElements;
}

void latticed2q9::step(void)
{
	stream();
	collide();
}

void latticed2q9::stream()
{
	//int i,j,im1,ip1,jm1,jp1,i0;

 //    //Initially the f's are moved to temporary arrays
	//for (j=0; j<_height; j++) 
	//{
	//	jm1=j-1;
	//	jp1=j+1;

	//	if (j==0) jm1=0;
	//	if (j==(_height-1)) jp1=_height-1;

	//	for (i=0; i<_width; i++) 
	//	{
	//		i0  = I2D(_width,i,j);
	//		im1 = i-1;
	//		ip1 = i+1;

	//		if (i==0) im1=0;
	//		if (i==(_width-1)) ip1=_width-1;

	//		latticeElements[i0].ftemp[1] = latticeElements[I2D(_width,im1,j)].f[1];
	//		latticeElements[i0].ftemp[2] = latticeElements[I2D(_width,i,jm1)].f[2];
	//		latticeElements[i0].ftemp[3] = latticeElements[I2D(_width,ip1,j)].f[3];
	//		latticeElements[i0].ftemp[4] = latticeElements[I2D(_width,i,jp1)].f[4];
	//		latticeElements[i0].ftemp[5] = latticeElements[I2D(_width,im1,jm1)].f[5];
	//		latticeElements[i0].ftemp[6] = latticeElements[I2D(_width,ip1,jm1)].f[6];
	//		latticeElements[i0].ftemp[7] = latticeElements[I2D(_width,ip1,jp1)].f[7];
	//		latticeElements[i0].ftemp[8] = latticeElements[I2D(_width,im1,jp1)].f[8];
	//	}
 //   }

 //   /* Now the temporary arrays are copied to the main f arrays*/
	//for (j=0; j<_height; j++) {
	//for (i=1; i<_width; i++) {
	//	i0 = I2D(_width,i,j);
	//    latticeElements[i0].f[1] = latticeElements[i0].ftemp[1];
	//	latticeElements[i0].f[2] = latticeElements[i0].ftemp[2];
	//	latticeElements[i0].f[3] = latticeElements[i0].ftemp[3];
	//	latticeElements[i0].f[4] = latticeElements[i0].ftemp[4];
	//	latticeElements[i0].f[5] = latticeElements[i0].ftemp[5];
	//	latticeElements[i0].f[6] = latticeElements[i0].ftemp[6];
	//	latticeElements[i0].f[7] = latticeElements[i0].ftemp[7];
	//	latticeElements[i0].f[8] = latticeElements[i0].ftemp[8];
	//}
 //   }

	int i0, i1;
	int newI, newJ;

	for(int j = 0; j < _height; j++)
	{
		for(int i = 0; i<_width; i++)
		{
			i0 =  I2D(_width, i,j);

			if(!latticeElements[i0].isSolid)
				for(int l = 0; l < 9; l++)
				{
					newI = (int)( i + speedDirection[l].x );
					newJ = (int)( j + speedDirection[l].y );
												
					////Checking for exit boundaries
					if(newI > (_width - 1)) newI = 0;
					else if(newI <= 0) newI = _width - 1;
						
					if(newJ > (_height - 1)) newJ = 0;
					else if(newJ <= 0) newJ = _height - 1;
						
					i1 =  I2D(_width, newI,newJ);
					latticeElements[i0].ftemp[l] = latticeElements[i1].f[l];
				}
		}
	}

	for(int i = 0; i < _numberElements; i++)
		for(int j = 0; j < 9; j++)
			latticeElements[i].f[j] = latticeElements[i].ftemp[j];
}

void latticed2q9::collide(void)
{
	int i0;
	float rtau = 1.0 / _tau, rtau1 = 1.f - rtau;

	for(int j = 0; j < _height; j++)
	{
		for(int i = 0; i<_width; i++)
		{
			i0 = I2D(_width, i, j);

			latticeElements[i0].calculateSpeedVector();
			latticeElements[i0].calculateEquilibriumFunction();

			if(!latticeElements[i0].isSolid)
			{
				for(int l = 0; l < 9; l++)
					latticeElements[i0].f[l] = latticeElements[i0].f[l] - ( latticeElements[i0].f[l] - latticeElements[i0].feq[l] ) / _tau;
					//latticeElements[i0].f[l] = rtau1 * latticeElements[i0].f[l] + rtau * latticeElements[i0].feq[l];
			}
			else
				solid_BC(i0);
		}
	}
}

void latticed2q9::solid_BC(int i0)
{
	double temp;

	temp = latticeElements[i0].f[1]; 	latticeElements[i0].f[1] = latticeElements[i0].f[3];	latticeElements[i0].f[3] = temp;		// f1	<-> f3
	temp = latticeElements[i0].f[2];	latticeElements[i0].f[2] = latticeElements[i0].f[4];	latticeElements[i0].f[4] = temp;				// f2	<-> f4
	temp = latticeElements[i0].f[5];	latticeElements[i0].f[5] = latticeElements[i0].f[7];	latticeElements[i0].f[7] = temp;				// f5	<-> f7
	temp = latticeElements[i0].f[6];	latticeElements[i0].f[6] = latticeElements[i0].f[8];	latticeElements[i0].f[8] = temp;				// f7	<-> f8

	//temp = latticeElements[i0].ftemp[1]; 	latticeElements[i0].ftemp[1] = latticeElements[i0].ftemp[3];	latticeElements[i0].ftemp[3] = temp;		// ftemp1	<-> ftemp3
	//temp = latticeElements[i0].ftemp[2];	latticeElements[i0].ftemp[2] = latticeElements[i0].ftemp[4];	latticeElements[i0].ftemp[4] = temp;				// ftemp2	<-> ftemp4
	//temp = latticeElements[i0].ftemp[5];	latticeElements[i0].ftemp[5] = latticeElements[i0].ftemp[7];	latticeElements[i0].ftemp[7] = temp;				// ftemp5	<-> ftemp7
	//temp = latticeElements[i0].ftemp[6];	latticeElements[i0].ftemp[6] = latticeElements[i0].ftemp[8];	latticeElements[i0].ftemp[8] = temp;				// f7	<-> f8

	//latticeElements[i0].f[0] = latticeElements[i0].ftemp[0];
	//latticeElements[i0].f[1] = latticeElements[i0].ftemp[1];
	//latticeElements[i0].f[2] = latticeElements[i0].ftemp[2];
	//latticeElements[i0].f[3] = latticeElements[i0].ftemp[3];
	//latticeElements[i0].f[4] = latticeElements[i0].ftemp[4];
	//latticeElements[i0].f[5] = latticeElements[i0].ftemp[5];
	//latticeElements[i0].f[6] = latticeElements[i0].ftemp[6];
	//latticeElements[i0].f[7] = latticeElements[i0].ftemp[7];
	//latticeElements[i0].f[8] = latticeElements[i0].ftemp[8];
}

void latticed2q9::printLattice(void)
{
	for(int i=0; i<_numberElements; i++)
	{
			latticeElements[i].printElement();
			std::cout << std::endl;
	}
}

void latticed2q9::printLatticeElement(int i, int j, int k)
{
	latticeElements[I3D(_width,_height, i,j,k)].printElement();
	std::cout << std::endl;
}

int latticed2q9::getNumElements(void)
{
	return _numberElements;
}