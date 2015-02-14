#include <iostream>

# ifndef DATATYPES
# define DATATYPES

//Function to linearly go over the arrays
#define I3D(width, depth,i,j,k) width*(j+depth*k)+i

template <class T>
class latticeElement
{
private:
	int _vectorVelocitiesSize;

public:
	T *velocities;

	latticeElement()
	{
	}

	latticeElement(int vectorVelocitiesSize)
	{
		_vectorVelocitiesSize = vectorVelocitiesSize;
		velocities = new T[vectorVelocitiesSize];
		for(int i=0; i<vectorVelocitiesSize; i++)
			velocities[i]=0;
	}

	void setSize(int vectorVelocitiesSize)
	{
		_vectorVelocitiesSize = vectorVelocitiesSize;
		velocities = new T[vectorVelocitiesSize];
		for(int i=0; i<vectorVelocitiesSize; i++)
			velocities[i]=0;
	}

	~latticeElement(void)
	{
		//free(velocities);
		std::cout << "destroying velocities" << std::endl;
		delete[] velocities;
	}

	void printElement(void)
	{
		for(int i =0; i<_vectorVelocitiesSize; i++)
			std::cout << velocities[i] << " ";
	}
};

template <class T> 
class lattice
{
private:
	int _width, _height, _depth, _numberElements;

public:
	latticeElement<T> *latticeElements;

	lattice()
	{
	}

	lattice(int width, int height, int depth, int velocities)
	{
		_width = width; _height = height; _depth = depth;
		_numberElements = _width*_height*_depth;
		latticeElements = new latticeElement<T>[_numberElements]; 
		for(int i = 0; i<_numberElements; i++)
			latticeElements[i].setSize(velocities);
	}

	~lattice()
	{
		delete[] latticeElements;
	}

	void printLattice(void)
	{
		for(int i=0; i<_numberElements; i++)
		{
				latticeElements[i].printElement();
				std::cout << std::endl;
		}
	}

	void printLatticeElement(int i, int j, int k)
	{
		std::cout << latticeElements[I3D(width,depth, i,j,k)].printElement() << std::endl;
	}

};
#endif