#pragma once
#ifndef M3_VECTOR_H
#define M3_VECTOR_H

//---------------------------------------------------------------------------

#include "m3Real.h"
#include <cassert>

//---------------------------------------------------------------------------
class m3Vector
	//---------------------------------------------------------------------------
{
public:
	inline m3Vector() { zero(); }
	inline m3Vector(const m3Vector& v0) :x(v0.x), y(v0.y), z(v0.z) { }//*this = v0; }
	inline m3Vector(m3Real x0, m3Real y0, m3Real z0) { x = x0; y = y0; z = z0; }
	inline void set(m3Real x0, m3Real y0, m3Real z0) { x = x0; y = y0; z = z0; }
	inline void zero() { x = 0.0; y = 0.0; z = 0.0; }
	inline bool isZero() { return x == 0.0 && y == 0.0 && z == 0.0; }

	m3Real & operator[] (int i) {
		assert(i >= 0 && i <= 2);
		return (&x)[i];
	}

	m3Vector& operator = (m3Vector rhs)
	{
		x = rhs.x; y = rhs.y; z = rhs.z;
		return *this;
	}

	bool operator == (const m3Vector &v) const {
		return (x == v.x) && (y == v.y) && (z ==v.z);
	}

	m3Vector operator + (const m3Vector &v) const {
		m3Vector r; 
		r.x = x + v.x; r.y = y + v.y; r.z = z + v.z;
		return r;
	}

	m3Vector operator - (const m3Vector &v) const {
		m3Vector r; r.x = x - v.x; r.y = y - v.y; r.z = z - v.z;
		return r;
	}
	void operator += (const m3Vector &v) {
		x += v.x; y += v.y; z += v.z;
	}
	void operator -= (const m3Vector &v) {
		x -= v.x; y -= v.y; z -= v.z;
	}
	void operator *= (const m3Vector &v) {
		x *= v.x; y *= v.y; z *= v.z;
	}
	void operator /= (const m3Vector &v) {
		x /= v.x; y /= v.y; z /= v.z;
	}
	m3Vector operator -() const {
		m3Vector r; 
		r.x = -x; r.y = -y; r.z = -z;
		return r;
	}
	m3Vector operator * (const m3Real f) const {
		m3Vector r; 
		r.x = x*f; r.y = y*f; r.z = z*f;
		return r;
	}
	m3Vector operator / (const m3Real f) const {
		m3Vector r;
		r.x = x / f; r.y = y / f; r.z = z / f;
		return r;
	}
	m3Vector cross(const m3Vector &v1, const m3Vector &v2) const {
		return m3Vector(v1.y*v2.z - v1.z *v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
	}

	inline m3Real dot(const m3Vector &v) const {
		return x*v.x + y*v.y + z*v.z;
	}

	inline void minimum(const m3Vector &v) {
		if (v.x < x) x = v.x;
		if (v.y < y) y = v.y;
		if (v.z < z) z = v.z;
	}
	inline void maximum(const m3Vector &v) {
		if (v.x > x) x = v.x;
		if (v.y > y) y = v.y;
		if (v.z > z) z = v.z;
	}

	inline m3Real magnitudeSquared() const { return x*x + y*y + z*z; }
	inline m3Real magnitude() const { return sqrt(x*x + y*y + z*z); }

	inline m3Real distanceSquared(const m3Vector &v) const {
		m3Real dx, dy, dz; 
		dx = v.x - x; dy = v.y - y; dz = v.z - z;
		return dx*dx + dy*dy + dz*dz;
	}

	inline m3Real distance(const m3Vector &v) const {
		m3Real dx, dy; dx = v.x - x; dy = v.y - y;
		return sqrt(dx*dx + dy*dy);
	}

	void operator *=(m3Real f) { x *= f; y *= f; z *= f; }
	void operator /=(m3Real f) { x /= f; y /= f; z /= f; }

	m3Vector normalize() 
	{
		m3Real l = magnitude();
		m3Vector v;
		if (l != 0.0f)
		{
			m3Real l1 = 1.0f / l; 
			x *= l1; y *= l1; z*=l1;
		}
	}

	// ------------------------------
	m3Real x, y, z;
};


#endif