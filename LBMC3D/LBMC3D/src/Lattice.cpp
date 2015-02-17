#include "Lattice.h"

latticed3q19::latticed3q19(int width, int height, int depth)
{
	_width = width; _height = height; _depth = depth;
	_numberElements = _width*_height*_depth;
	latticeElements = new latticeElementd3q19[_numberElements]; 
	tempLatticeElements = new latticeElementd3q19[_numberElements]; 
}
	
latticed3q19::~latticed3q19()
{
	delete[] latticeElements;
	delete[] tempLatticeElements;
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

				for(int l = 0; l < 19; l++)
				{
					newI = i + speedDirection[l][0];
					newJ = j + speedDirection[l][1];
					newK = k + speedDirection[l][2];
					
					i1 = I3D(_width, _height, newI > 0?newI:0, newJ >0?newJ:0, newK>0?newK:0);
					tempLatticeElements[i0] = latticeElements[i1];
				}
			}
		}
	}

	for(int k =0; k<_depth; k++)
	{
		for(int j = 0; j < _height; j++)
		{
			for(int i = 0; i<_width; i++)
			{
				i0 = I3D(_width, _height, i, j, k);

				for(int l = 0; l < 19; l++)
					latticeElements[i0] = tempLatticeElements[i0];
			}
		}
	}
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