#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth, float tau)
{
	_width = width; _height = height; _depth = depth; _tau = tau;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd3q19[_numberElements];
}
	
latticed3q19::~latticed3q19()
{
	delete[] latticeElements;
}

void latticed3q19::step(void)
{
	stream();
	collide();
}

void latticed3q19::stream()
{
	int i0, i1;
	int newI, newJ, newK;

	for(int k =0; k<_depth; k++)
	{
		for(int j = 0; j < _height; j++)
		{
			for(int i = 0; i<_width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				if(!latticeElements[i0].isSolid)
					for(int l = 0; l < 19; l++)
					{
						newI = (int)( i + speedDirection[l].x );
						newJ = (int)( j + speedDirection[l].y );
						newK = (int)( k + speedDirection[l].z );
						
						//Checking for exit boundaries
						if(newI > (_width - 1)) newI = 0;
						else if(newI <= 0) newI = _width - 1;
						
						if(newJ > (_height - 1)) newJ = 0;
						else if(newJ <= 0) newJ = _height - 1;

						if(newK > (_depth - 1)) newK = 0;
						else if(newK <= 0) newK = _depth - 1;

						//if(newI> (_width-1) || newI<=0) 
						//	newI = (int)( i - speedDirection[l].x );
						//if(newJ> (_height-1) || newJ<=0) 
						//	newJ = (int)( j - speedDirection[l].y );
						//if(newK> (_depth-1) || newK<=0) 
						//	newK = (int)( k - speedDirection[l].z );

						i1 = I3D(_width, _height, newI, newJ , newK );

						latticeElements[i0].ftemp[l] = latticeElements[i1].f[l];
					}
			}
		}
	}

	for(int i = 0; i < _numberElements; i++)
		for(int j = 0; j < 19; j++)
			latticeElements[i].f[j] = latticeElements[i].ftemp[j];
}

void latticed3q19::collide(void)
{
	int i0;

	for(int k =0; k<_depth; k++)
	{
		for(int j = 0; j < _height; j++)
		{
			for(int i = 0; i<_width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				latticeElements[i0].calculateSpeedVector();
				latticeElements[i0].calculateEquilibriumFunction();

				if(!latticeElements[i0].isSolid)
				{
					for(int l = 0; l < 19; l++)
						latticeElements[i0].f[l] = latticeElements[i0].f[l] - ( latticeElements[i0].f[l] - latticeElements[i0].feq[l] ) / _tau;
						//latticeElements[i0].f[l] =(1-_tau)* latticeElements[i0].ftemp[l] + (1/_tau) * latticeElements[i0].feq[l];
				}
				else
					solid_BC(i0);
			}
		}
	}
}

void latticed3q19::solid_BC(int i0)
{
	double temp;

	temp = latticeElements[i0].f[1]; 	latticeElements[i0].f[1] = latticeElements[i0].f[3];	latticeElements[i0].f[3] = temp;		// f1	<-> f3
	temp = latticeElements[i0].f[2];	latticeElements[i0].f[2] = latticeElements[i0].f[4];	latticeElements[i0].f[4] = temp;		// f2	<-> f4
	temp = latticeElements[i0].f[5];	latticeElements[i0].f[5] = latticeElements[i0].f[6];	latticeElements[i0].f[6] = temp;		// f5	<-> f6
	temp = latticeElements[i0].f[7];	latticeElements[i0].f[7] = latticeElements[i0].f[9];	latticeElements[i0].f[9] = temp;		// f7	<-> f9
	temp = latticeElements[i0].f[8];	latticeElements[i0].f[8] = latticeElements[i0].f[10];	latticeElements[i0].f[10] = temp;		// f8	<-> f10
	temp = latticeElements[i0].f[11];	latticeElements[i0].f[11] = latticeElements[i0].f[13];	latticeElements[i0].f[13] = temp;		// f11	<-> f13
	temp = latticeElements[i0].f[12];	latticeElements[i0].f[12] = latticeElements[i0].f[14];	latticeElements[i0].f[14] = temp;		// f12	<-> f14
	temp = latticeElements[i0].f[15];	latticeElements[i0].f[15] = latticeElements[i0].f[18];	latticeElements[i0].f[18] = temp;		// f15	<-> f18
	temp = latticeElements[i0].f[16];	latticeElements[i0].f[16] = latticeElements[i0].f[17];	latticeElements[i0].f[17] = temp;		// f16	<-> f17

}

void latticed3q19::printLattice(void)
{
	for(int i=0; i<_numberElements; i++)
	{
			latticeElements[i].printElement();
			std::cout << std::endl;
	}
}

void latticed3q19::printLatticeElement(int i, int j, int k)
{
	latticeElements[I3D(_width,_height, i,j,k)].printElement();
	std::cout << std::endl;
}

int latticed3q19::getNumElements(void)
{
	return _numberElements;
}