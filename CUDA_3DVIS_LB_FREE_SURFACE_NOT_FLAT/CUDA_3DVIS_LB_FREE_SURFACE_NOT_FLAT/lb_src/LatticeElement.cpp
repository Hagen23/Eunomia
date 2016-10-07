#include "LatticeElement.h"

latticeElementd3q19::latticeElementd3q19()
{
	_vectorVelocitiesSize = 19;
		
	c = 1.0f / sqrtf(3.0);
	f = new float[_vectorVelocitiesSize]();
	feq = new float[_vectorVelocitiesSize]();
	ftemp = new float[_vectorVelocitiesSize]();

	velocityVector = { 0.0f, 0.0f, 0.0f };
		
	ro = 0;
	rovx = rovy = rovz = 0;
	cellMass = 0;
	cellMassTemp = 0;
	epsilon = 0;
	cellType = cellTypeTemp = cell_types::gas;
	isSolid = false;
}
	
latticeElementd3q19::~latticeElementd3q19(void)
{
	//std::cout << "destroying velocities" << std::endl;
	//delete[] f;
	//delete[] feq;
	//delete[] ftemp;
}

void latticeElementd3q19::printElement(void)
{
	for(int i =0; i<_vectorVelocitiesSize; i++)
		std::cout << "f[" << i << "]: " << f[i] << " vx: " << velocityVector.x << " vy: " << velocityVector.y << " vz: " << velocityVector.z << std::endl;
}

float latticeElementd3q19::getEpsilon(void)
{
	if (ro > 0)
	{
		epsilon = cellMass / ro;
		return epsilon;
	}
	return 0;
}

void latticeElementd3q19::calculateQuantities(void)
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
	if (ro > 0)
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

void latticeElementd3q19::calculateEquilibriumFunction(vector3d inVector, float inRo)
{
	float w;
	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
	float uSq = inVector.dotProduct(inVector);	//Velocity squared

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		w = latticeWeights[i];
		eiU = speedDirection[i].dotProduct(inVector);
		eiUsq = eiU * eiU;

		//feq[i] = w * ro * ( 1.f + 3.f * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		feq[i] = w * inRo * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}
}

void latticeElementd3q19::calculateInEquilibriumFunction(vector3d inVector, float inRo)
{
	float w;

	float eiU = 0;	// Dot product between speed direction and velocity
	float eiUsq = 0; // Dot product squared
		
	float uSq = inVector.dotProduct(inVector);	//Velocity squared

	//ro = inRo;

	for(int i=0; i<_vectorVelocitiesSize; i++)
	{
		w = latticeWeights[i];
		eiU = speedDirection[i].dotProduct(inVector);
		eiUsq = eiU * eiU;

		//ftemp[i] = f[i] = w * ro * ( 1 + 3 * (eiU) + 4.5 * (eiUsq) -1.5 * (uSq));
		ftemp[i] = f[i] = w;// *ro * (1.f + (eiU) / (c*c) + (eiUsq) / (2 * c * c * c * c) - (uSq) / (2 * c * c));
	}

	calculateQuantities();

	if ((cellType & cell_types::fluid) == cell_types::fluid)
		cellMass = cellMassTemp = ro; // 1.0f;
	else if ((cellType & cell_types::interphase) == cell_types::interphase)
		cellMass = cellMassTemp = 0.1f * ro;
	else if ((cellType & cell_types::gas) == cell_types::gas)
		cellMass = cellMassTemp = 0.0f;

	getEpsilon();
	int test = 0;
}