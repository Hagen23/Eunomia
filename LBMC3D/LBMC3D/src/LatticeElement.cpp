#include "LatticeElement.h"

void latticeElementd3q19::calculateRo(void)
{
	ro = 0; 
	for(int i = 0; i<_vectorVelocitiesSize; i++)
		ro += f[i];
}


latticeElementd3q19::latticeElementd3q19()
{
	_vectorVelocitiesSize = 19;
		
	f = new double[_vectorVelocitiesSize];
	feq = new double[_vectorVelocitiesSize];
	ftemp = new double[_vectorVelocitiesSize];

	velocityVector.x = 1;
	velocityVector.y = 1;
	velocityVector.z = 1;
		
	isSolid = false;

	for(int i = 0; i<_vectorVelocitiesSize; i++)
	{
		f[i] = 0;
		feq[i] = 0;
	}
}
	
latticeElementd3q19::~latticeElementd3q19(void)
{
	std::cout << "destroying velocities" << std::endl;
	delete[] f;
	delete[] feq;
}

void latticeElementd3q19::printElement(void)
{
	for(int i =0; i<_vectorVelocitiesSize; i++)
		std::cout << "f[" << i << "]: " << f[i] << " vx: " << velocityVector.x << " vy: " << velocityVector.y << " vz: " << velocityVector.z << std::endl;
}

void latticeElementd3q19::calculateSpeedVector(void)
{
	calculateRo();
	rovx = rovy = rovz = 0; 

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		rovx += f[i] * speedDirection[i].x;
		rovy += f[i] * speedDirection[i].y;
		rovz += f[i] * speedDirection[i].z;
	}

	velocityVector.x = rovx / ro;
	velocityVector.y = rovy / ro;
	velocityVector.z = rovz / ro;
}
	
void latticeElementd3q19::calculateEquilibriumFunction()
{
	double w;

	double eiU = 0;	// Dot product between speed direction and velocity
	double eiUsq = 0; // Dot product squared
		
	double uSq = velocityVector.dotProduct(velocityVector);	//Velocity squared

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		if(i == 0) w = 1.0/3.0;
		if(i >=1 && i <=6) w = 1.0/18.0;
		if(i >=7 && i <=18) w = 1.0/36.0;

		eiU = speedDirection[i].dotProduct(velocityVector);
		eiU = eiU * eiU;

		feq[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
	}
}