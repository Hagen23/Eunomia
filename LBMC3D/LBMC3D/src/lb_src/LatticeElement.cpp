#include "LatticeElement.h"

latticeElementd2q9::latticeElementd2q9()
{
	_vectorVelocitiesSize = 9;
		
	f = new double[_vectorVelocitiesSize];
	feq = new double[_vectorVelocitiesSize];
	ftemp = new double[_vectorVelocitiesSize];

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
	
latticeElementd2q9::~latticeElementd2q9(void)
{
	std::cout << "destroying velocities" << std::endl;
	delete[] f;
	delete[] feq;
}

void latticeElementd2q9::printElement(void)
{
	for(int i =0; i<_vectorVelocitiesSize; i++)
		std::cout << "f[" << i << "]: " << f[i] << " vx: " << velocityVector.x << " vy: " << velocityVector.y << " vz: " << velocityVector.z << std::endl;
}

void latticeElementd2q9::calculateSpeedVector(void)
{
	ro = rovx = rovy = rovz = 0; 

	//ro = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
 //   rovx = f[1] - f[3] + f[5] - f[6] - f[7] + f[8];
 //   rovy = f[2] - f[4] + f[5] + f[6] - f[7] - f[8];

	//ro = rovx = rovy = rovz = 0; 

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		ro += f[i];
		rovx += f[i] * speedDirection[i].x;
		rovy += f[i] * speedDirection[i].y;
	}

	// In order to check that ro is not NaN you check if it is equal to itself: if it is a Nan, the comparison is false
	if(ro ==ro && ro != 0.0)
	{
		velocityVector.x = rovx / ro;
		velocityVector.y = rovy / ro;
		velocityVector.z = 0;
	}
}
	
void latticeElementd2q9::calculateEquilibriumFunction()
{
	double w;
	double eiU = 0;	// Dot product between speed direction and velocity
	double eiUsq = 0; // Dot product squared
		
	double uSq = velocityVector.dotProduct(velocityVector);	//Velocity squared

	float vx = velocityVector.x, vy = velocityVector.y;
	float v_sq_term = 1.5f*(vx*vx + vy*vy);

	//feq[0] = ro * 4.0/9.0 * (1.f - v_sq_term);
	//feq[1] = ro * 1.0/9.0 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
	//feq[2] = ro * 1.0/9.0 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
	//feq[3] = ro * 1.0/9.0 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
	//feq[4] = ro * 1.0/9.0 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
	//feq[5] = ro * 1.0/36.0 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
	//feq[6] = ro * 1.0/36.0 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
	//feq[7] = ro * 1.0/36.0 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
	//feq[8] = ro * 1.0/36.0 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);
	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		if(i == 0) w = 4.0/9.0; else
		if(i >=1 && i <=4) w = 1.0/9.0; else
		if(i >=5 && i <=8) w = 1.0/36.0;

		eiU = speedDirection[i].dotProduct(velocityVector);
		eiUsq = eiU * eiU;

		feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) - 1.5 * (uSq));
	}
}

void latticeElementd2q9::calculateInEquilibriumFunction(vector3d inVector, float inRo)
{
	double w;

	double eiU = 0;	// Dot product between speed direction and velocity
	double eiUsq = 0; // Dot product squared
		
	double uSq = inVector.dotProduct(inVector);	//Velocity squared

	ro = inRo;

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		if(i == 0) w = 4.0/9.0; else
		if(i >=1 && i <=4) w = 1.0/9.0; else
		if(i >=5 && i <=8) w = 1.0/36.0;

		eiU = speedDirection[i].dotProduct(inVector);
		eiUsq = eiU * eiU;

		ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
	}
}