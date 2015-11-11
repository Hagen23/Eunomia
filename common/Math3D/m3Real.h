#pragma once
#ifndef M3_REAL_H
#define M3_REAL_H

#include <cmath>
#include <cstdlib>
#include <cfloat>

#define m3Pi     3.1415926535897932f
#define m3HalfPi 1.5707963267948966f
#define m3TwoPi  6.2831853071795865f
#define m3RealMax FLT_MAX
#define m3RealMin FLT_MIN
#define m3RadToDeg 57.295779513082321f
#define m3DegToRad 0.0174532925199433f 

typedef float m3Real;

//---------------------------------------------------------------------------

inline m3Real m2Clamp(m3Real &r, m3Real min, m3Real max)
{
	if (r < min) return min;
	else if (r > max) return max;
	else return r;
}

//---------------------------------------------------------------------------

inline m3Real m2Min(m3Real r1, m3Real r2)
{
	if (r1 <= r2) return r1;
	else return r2;
}

//---------------------------------------------------------------------------

inline m3Real m2Max(m3Real r1, m3Real r2)
{
	if (r1 >= r2) return r1;
	else return r2;
}

//---------------------------------------------------------------------------

inline m3Real m2Abs(m3Real r)
{
	if (r < 0.0f) return -r;
	else return r;
}

//---------------------------------------------------------------------------

inline m3Real m2Random(m3Real min, m3Real max)
{
	return min + ((m3Real)rand() / RAND_MAX) * (max - min);
}

//---------------------------------------------------------------------------

inline m3Real m2Acos(m3Real r)
{
	// result is between 0 and pi
	if (r < -1.0f) r = -1.0f;
	if (r >  1.0f) r = 1.0f;
	return acos(r);
}

#endif

