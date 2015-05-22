#include "LatticeElement.h"

latticeElementd3q19::latticeElementd3q19()
{
	_vectorVelocitiesSize = 19;
		
	c = 1.0 / sqrt(3.0);
	f = new float[_vectorVelocitiesSize];
	feq = new float[_vectorVelocitiesSize];
	ftemp = new float[_vectorVelocitiesSize];

	velocityVector.x = 0;
	velocityVector.y = 0;
	velocityVector.z = 0;
		
	isSolid = false;

	for(int i = 0; i<_vectorVelocitiesSize; i++)
	{
		f[i] = 0;
		feq[i] = 0;
		ftemp[i] = 0;
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
//
//void latticeElementd3q19::calculateRo(void)
//{
//	ro = 0; 
//	for(int i = 0; i<_vectorVelocitiesSize; i++)
//		ro += f[i];
//}

void latticeElementd3q19::calculateSpeedVector(void)
{
	//calculateRo();
	//rovx = rovy = rovz = 0; 

	ro = rovx = rovy = rovz = 0; 

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		ro += f[i];
		rovx += f[i] * speedDirection[i].x;
		rovy += f[i] * speedDirection[i].y;
		rovz += f[i] * speedDirection[i].z;
	}

	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
	if(ro ==ro && ro != 0.0)
	{
		velocityVector.x = rovx / ro;
		velocityVector.y = rovy / ro;
		velocityVector.z = rovz / ro;
	}
	else
	{
		velocityVector.x = 0;
		velocityVector.y = 0;
		velocityVector.z = 0;
	}
}
	
void latticeElementd3q19::calculateEquilibriumFunction()
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = velocityVector.dotProduct(velocityVector);	//Velocity squared

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		w = latticeWeights[i];
		eiU = speedDirection[i].dotProduct(velocityVector);
		eiUsq = eiU * eiU;

		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		feq[i] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

void latticeElementd3q19::calculateInEquilibriumFunction(vector3d inVector, float inRo)
{
	float w;

	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
		
	float uSq = inVector.dotProduct(inVector);	//Velocity squared

	ro = inRo;

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		w = latticeWeights[i];
		eiU = speedDirection[i].dotProduct(inVector);
		eiUsq = eiU * eiU;

		//ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		ftemp[i] = f[i] = w * ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}
