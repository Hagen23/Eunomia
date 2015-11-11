
#include "deformable.h"
#include <stdio.h>

//---------------------------------------------------------------------------
void DeformableParameters::setDefaults()
{
	timeStep = 0.01f;
	gravity.set(0.0f, -9.81f, 0.0f);

	bounds.min.zero();
	bounds.max.set(1.0f, 1.0f, 1.0f);

	alpha = 0.05f;
	beta = 0.60f;

	quadraticMatch = true;
	volumeConservation = true;

	allowFlip = true;
}

//---------------------------------------------------------------------------
Deformable::Deformable()
{
	reset();
}

//---------------------------------------------------------------------------
Deformable::~Deformable()
{
}


//---------------------------------------------------------------------------
void Deformable::reset()
{
	mNumVertices = 0;
	mOriginalPos.clear();
	mPos.clear();
	mNewPos.clear();
	mGoalPos.clear();
	mMasses.clear();
	mVelocities.clear();
	mFixed.clear();
}

//---------------------------------------------------------------------------
void Deformable::initState()
{
	for (int i = 0; i < mNumVertices; i++) 
	{
		mPos[i] = mOriginalPos[i];
		mNewPos[i] = mOriginalPos[i];
		mGoalPos[i] = mOriginalPos[i];
		mVelocities[i].zero();
		mFixed[i] = false;
	}
}

//---------------------------------------------------------------------------
void Deformable::addVertex(const m3Vector &pos, float mass)
{
	mOriginalPos.push_back(pos);
	mPos.push_back(pos);
	mNewPos.push_back(pos);
	mGoalPos.push_back(pos);
	mMasses.push_back(mass);
	mVelocities.push_back(m3Vector(0, 0, 0));
	mFixed.push_back(false);
	mNumVertices++;

	initState();
}

//---------------------------------------------------------------------------
void Deformable::externalForces()
{
	int i;

	for (i = 0; i < mNumVertices; i++) 
	{
		if (mFixed[i]) continue;
		mVelocities[i] += params.gravity * params.timeStep;
		mNewPos[i] = mPos[i] + mVelocities[i] * params.timeStep;
		mGoalPos[i] = mOriginalPos[i];
	}

	// boundaries
	m3Real restitution = 0.9f;
	for (i = 0; i < mNumVertices; i++) 
	{
		if (mFixed[i]) continue;
		m3Vector &p = mPos[i];
		m3Vector &np = mNewPos[i];
		m3Vector &v = mVelocities[i];
		if (np.x < params.bounds.min.x || np.x > params.bounds.max.x) {
			np.x = p.x - v.x * params.timeStep * restitution;
			np.y = p.y;
			np.z = p.z;
		}
		if (np.y < params.bounds.min.y || np.y > params.bounds.max.y) {
			np.y = p.y - v.y * params.timeStep * restitution;
			np.x = p.x;
			np.z = p.z;
		}
		if (np.z< params.bounds.min.z || np.z > params.bounds.max.z) {
			np.z = p.z - v.z * params.timeStep * restitution;
			np.y = p.y;
			np.x = p.x;
		}
		params.bounds.clamp(mNewPos[i]);
	}

}
//---------------------------------------------------------------------------
void Deformable::projectPositions()
{
	if (mNumVertices <= 1) return;
	int i, j, k;

	// center of mass
	m3Vector cm, originalCm;
	cm.zero(); originalCm.zero();
	float mass = 0.0f;

	for (i = 0; i < mNumVertices; i++) 
	{
		m3Real m = mMasses[i];
		if (mFixed[i]) m *= 100.0f;
		mass += m;
		cm += mNewPos[i] * m;
		originalCm += mOriginalPos[i] * m;
	}

	cm /= mass;
	originalCm /= mass;

	m3Matrix Apq, Aqq;
	m3Vector p, q;
	Apq.zero();
	Aqq.zero();

	for (i = 0; i < mNumVertices; i++) 
	{
		p = mNewPos[i] - cm;
		q = mOriginalPos[i] - originalCm;
		m3Real m = mMasses[i];
		Apq.r00 += m * p.x * q.x;
		Apq.r01 += m * p.x * q.y;
		Apq.r02 += m * p.x * q.z;

		Apq.r10 += m * p.y * q.x;
		Apq.r11 += m * p.y * q.y;
		Apq.r12 += m * p.y * q.z;

		Apq.r20 += m * p.z * q.x;
		Apq.r21 += m * p.z * q.y;
		Apq.r22 += m * p.z * q.z;
		
		Aqq.r00 += m * q.x * q.x;
		Aqq.r01 += m * q.x * q.y;
		Aqq.r02 += m * q.x * q.z;

		Aqq.r10 += m * q.y * q.x;
		Aqq.r11 += m * q.y * q.y;
		Aqq.r12 += m * q.y * q.z;

		Aqq.r20 += m * q.z * q.x;
		Aqq.r21 += m * q.z * q.y;
		Aqq.r22 += m * q.z * q.z;
	}

	if (!params.allowFlip && Apq.determinant() < 0.0f) 
	{  	// prevent from flipping
		Apq.r01 = -Apq.r01;
		Apq.r11 = -Apq.r11;
		Apq.r22 = -Apq.r22;
	}

	m3Matrix R, S;
	m3Matrix::polarDecomposition(Apq, R, S);

	if (!params.quadraticMatch) 
	{	// --------- linear match
		m3Matrix A = Aqq;
		A.invert();
		A.multiply(Apq, A);

		if (params.volumeConservation)
		{
			m3Real det = A.determinant();
			if (det != 0.0f) 
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;
				A *= det;
			}
		}

		m3Matrix T = R * (1.0f - params.beta) + A * params.beta;

		for (i = 0; i < mNumVertices; i++) 
		{
			if (mFixed[i]) continue;
			q = mOriginalPos[i] - originalCm;
			mGoalPos[i] = T.multiply(q) + cm;
			mNewPos[i] += (mGoalPos[i] - mNewPos[i]) * params.alpha;
		}
	}
	else 
	{	// -------------- quadratic match---------------------

		m3Real A9pq[3][9];

		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 9; j++)
			A9pq[i][j] = 0.0f;

		m9Matrix A9qq;
		A9qq.zero();

		for (i = 0; i < mNumVertices; i++) 
		{
			p = mNewPos[i] - cm;
			q = mOriginalPos[i] - originalCm;

			m3Real q9[9];
			q9[0] = q.x; q9[1] = q.y; q9[2] = q.z; q9[3] = q.x*q.x; q9[4] = q.y*q.y; q9[5] = q.z*q.z;
			q9[6] = q.x*q.y; q9[7] = q.y*q.z; q9[8] = q.z*q.x; 

			m3Real m = mMasses[i];
			A9pq[0][0] += m * p.x * q9[0];
			A9pq[0][1] += m * p.x * q9[1];
			A9pq[0][2] += m * p.x * q9[2];
			A9pq[0][3] += m * p.x * q9[3];
			A9pq[0][4] += m * p.x * q9[4];
			A9pq[0][5] += m * p.x * q9[5];
			A9pq[0][6] += m * p.x * q9[6];
			A9pq[0][7] += m * p.x * q9[7];
			A9pq[0][8] += m * p.x * q9[8];

			A9pq[1][0] += m * p.y * q9[0];
			A9pq[1][1] += m * p.y * q9[1];
			A9pq[1][2] += m * p.y * q9[2];
			A9pq[1][3] += m * p.y * q9[3];
			A9pq[1][4] += m * p.y * q9[4];
			A9pq[1][5] += m * p.y * q9[5];
			A9pq[1][6] += m * p.y * q9[6];
			A9pq[1][7] += m * p.y * q9[7];
			A9pq[1][8] += m * p.y * q9[8];

			A9pq[2][0] += m * p.z * q9[0];
			A9pq[2][1] += m * p.z * q9[1];
			A9pq[2][2] += m * p.z * q9[2];
			A9pq[2][3] += m * p.z * q9[3];
			A9pq[2][4] += m * p.z * q9[4];
			A9pq[2][5] += m * p.z * q9[5];
			A9pq[2][6] += m * p.z * q9[6];
			A9pq[2][7] += m * p.z * q9[7];
			A9pq[2][8] += m * p.z * q9[8];

			for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				A9qq(j, k) += m * q9[j] * q9[k];
		}

		A9qq.invert();

		m3Real A9[3][9];
		for (i = 0; i < 3; i++) 
		{
			for (j = 0; j < 9; j++) 
			{
				A9[i][j] = 0.0f;
				for (k = 0; k < 9; k++) 
					A9[i][j] += A9pq[i][k] * A9qq(k, j);
				
				A9[i][j] *= params.beta;
				if (j < 3)
					A9[i][j] += (1.0f - params.beta) * R(i, j);
			}
		}

		m3Real det = 
			A9[0][0] * (A9[1][1] * A9[2][2] - A9[2][1] * A9[1][2]) -
			A9[0][1] * (A9[1][0] * A9[2][2] - A9[2][0] * A9[1][2]) + 
			A9[0][2] * (A9[1][0] * A9[2][1] - A9[1][1] * A9[2][0]);

		if (!params.allowFlip && det < 0.0f) {         		// prevent from flipping
			A9[0][1] = -A9[0][1];
			A9[1][1] = -A9[1][1];
			A9[2][2] = -A9[2][2];
		}

		if (params.volumeConservation) 
		{
			if (det != 0.0f) 
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;

				for (int i = 0; i < 3; i++)
				for (int j = 0; j < 9; j++)
					A9[i][j] *= det;
			}
		}
		
		for (i = 0; i < mNumVertices; i++)
		{
			if (mFixed[i]) continue;
			q = mOriginalPos[i] - originalCm;

			mGoalPos[i].x = A9[0][0] * q.x + A9[0][1] * q.y + A9[0][2] * q.z + A9[0][3] * q.x*q.x + A9[0][4] * q.y*q.y +
				A9[0][5] * q.z*q.z + A9[0][6] * q.x*q.y + A9[0][7] * q.y*q.z + A9[0][8] * q.z*q.x;

			mGoalPos[i].y = A9[1][0] * q.x + A9[1][1] * q.y + A9[1][2] * q.z + A9[1][3] * q.x*q.x + A9[1][4] * q.y*q.y +
				A9[1][5] * q.z*q.z + A9[1][6] * q.x*q.y + A9[1][7] * q.y*q.z + A9[1][8] * q.z*q.x;
			
			mGoalPos[i].z = A9[2][0] * q.x + A9[2][1] * q.y + A9[2][2] * q.z + A9[2][3] * q.x*q.x + A9[2][4] * q.y*q.y +
				A9[2][5] * q.z*q.z + A9[2][6] * q.x*q.y + A9[2][7] * q.y*q.z + A9[2][8] * q.z*q.x;

			mGoalPos[i] += cm;
			mNewPos[i] += (mGoalPos[i] - mNewPos[i]) * params.alpha;
		}
	}
}

//---------------------------------------------------------------------------
void Deformable::integrate()
{
	m3Real dt1 = 1.0f / params.timeStep;
	for (int i = 0; i < mNumVertices; i++) 
	{
		mVelocities[i] = (mNewPos[i] - mPos[i]) * dt1;
		mPos[i] = mNewPos[i];
	}
}

//---------------------------------------------------------------------------
void Deformable::timeStep()
{
	externalForces();
	projectPositions();
	integrate();
}


//---------------------------------------------------------------------------
void Deformable::fixVertex(int nr, const m3Vector &pos)
{
	mNewPos[nr] = pos;
	mFixed[nr] = true;
}


//---------------------------------------------------------------------------
void Deformable::releaseVertex(int nr)
{
	mFixed[nr] = false;
}

//---------------------------------------------------------------------------
void Deformable::saveToFile(char *filename)
{
	FILE *f = fopen(filename, "w");
	if (!f) return;

	fprintf(f, "%i\n", mNumVertices);
	for (int i = 0; i < mNumVertices; i++) {
		fprintf(f, "%f %f %f\n", mOriginalPos[i].x, mOriginalPos[i].y, mMasses[i]);
	}
	fprintf(f, "%f\n", params.timeStep);
	fprintf(f, "%f %f\n", params.gravity.x, params.gravity.y);

	fprintf(f, "%f\n", params.alpha);
	fprintf(f, "%f\n", params.beta);

	fprintf(f, "%i\n", params.quadraticMatch);
	fprintf(f, "%i\n", params.volumeConservation);
	fprintf(f, "%i\n", params.allowFlip);

	fclose(f);
}


//---------------------------------------------------------------------------
void Deformable::loadFromFile(char *filename)
{
	FILE *f = fopen(filename, "r");
	if (!f) return;

	const int len = 100;
	char s[len + 1];
	m3Vector pos;
	m3Real mass;
	int i;

	reset();
	int numVerts;
	fgets(s, len, f); sscanf(s, "%i", &numVerts);

	for (i = 0; i < numVerts; i++) {
		fgets(s, len, f); sscanf(s, "%f %f %f", &pos.x, &pos.y, &mass);
		addVertex(pos, mass);
	}

	fgets(s, len, f); sscanf(s, "%f", &params.timeStep);
	fgets(s, len, f); sscanf(s, "%f %f", &params.gravity.x, &params.gravity.y);

	fgets(s, len, f); sscanf(s, "%f", &params.alpha);
	fgets(s, len, f); sscanf(s, "%f", &params.beta);

	fgets(s, len, f); sscanf(s, "%i", &i); params.quadraticMatch = i >=1?1:0;
	fgets(s, len, f); sscanf(s, "%i", &i); params.volumeConservation = i >= 1 ? 1 : 0;
	fgets(s, len, f); sscanf(s, "%i", &i); params.allowFlip = i >= 1 ? 1 : 0;

	fclose(f);
}

