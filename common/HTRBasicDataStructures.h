/**
*@file HTRBasicDataStructures.h
*Data structures that do not depend on external classes.
*
* @author  Jonathan Langford
*
*/

#pragma once
#ifndef HTR_BASIC_DATA_STRUCTURES_H
#define HTR_BASIC_DATA_STRUCTURES_H

#include <cstdlib>

namespace htr{
	struct Index2D{
		int x;
		int y;
		//Index2D():x(0),y(0){}
	};

	struct Point3D{

		enum point_type { inside, boundary, air };

		float x;
		float y;
		float z;
		
		point_type type;

		Point3D() :x(0), y(0), z(0), type(point_type::inside)
		{}

		Point3D(float x0, float y0, float z0, point_type _type = point_type::inside) :x(x0), y(y0), z(z0), type(_type)
		{}

		Point3D& operator+=(const Point3D& pointToAdd)
		{
			this->x += pointToAdd.x;
			this->y += pointToAdd.y;
			this->z += pointToAdd.z;
			return *this;
		}

		Point3D& operator+=(const float& pointToAdd)
		{
			this->x += pointToAdd;
			this->y += pointToAdd;
			this->z += pointToAdd;
			return *this;
		}

		Point3D& operator-=(const float& pointToAdd)
		{
			this->x -= pointToAdd;
			this->y -= pointToAdd;
			this->z -= pointToAdd;
			return *this;
		}

		Point3D& operator-=(const Point3D& pointToAdd)
		{
			this->x -= pointToAdd.x;
			this->y -= pointToAdd.y;
			this->z -= pointToAdd.z;
			return *this;
		}

		Point3D& operator*=(const Point3D& pointToDivide)
		{
			this->x *= pointToDivide.x;
			this->y *= pointToDivide.y;
			this->z *= pointToDivide.z;
			return *this;
		}

		Point3D& operator/=(const Point3D& pointToDivide)
		{
			this->x /= pointToDivide.x;
			this->y /= pointToDivide.y;
			this->z /= pointToDivide.z;
			return *this;
		}

		Point3D& operator/=(const unsigned int& valueToDivide)
		{
			this->x /= valueToDivide;
			this->y /= valueToDivide;
			this->z /= valueToDivide;
			return *this;
		}

		Point3D& operator*=(const float& valueToMultiply)
		{
			this->x *= valueToMultiply;
			this->y *= valueToMultiply;
			this->z *= valueToMultiply;
			return *this;
		}

		Point3D& operator/=(const float& valueToDivide)
		{
			this->x /= valueToDivide;
			this->y /= valueToDivide;
			this->z /= valueToDivide;
			return *this;
		}

		bool operator== (const Point3D &v) const
		{
			return (
				(this->x == v.x) &&
				(this->y == v.y) &&
				(this->z == v.z));
		}

		bool operator<(const Point3D& rhs) const
		{
			return (y < rhs.y);//(x < rhs.x) && (y < rhs.y) && (z < rhs.z);
		}

		void initRandom()
        {
            x = float(rand() % 40);
			y = float(rand() % 40);
			z = float(rand() % 40);
        }

		void scalePoint(float scale)
		{
			*this *= scale;
		}

		void clear()
		{
			x = 0;
			y = 0;
			z = 0;
		}
	};

	inline Point3D operator+(Point3D lhs, const Point3D& rhs)
	{
		return lhs += rhs;
	}

	inline Point3D operator-(Point3D lhs, const Point3D& rhs)
	{
		return lhs -= rhs;
	}

	inline Point3D operator/(Point3D lhs, const Point3D& rhs)
	{
		return lhs /= rhs;
	}

	inline Point3D operator*(Point3D lhs, const Point3D& rhs)
	{
		return lhs *= rhs;
	}

	inline Point3D operator*(Point3D lhs, const float& rhs)
	{
		return lhs *= rhs;
	}

	inline Point3D operator/(Point3D lhs, const float& rhs)
	{
		return lhs /= rhs;
	}

	struct FlaggedPoint3D{
		Point3D point;
		int flag;
	};

	struct DepthPixel{
		int x;
		int y;
		float z;
	};

	struct LabeledPoint{
		Point3D point;
		int label;
	};

	struct CubeBoundary{
		Point3D start;
		Point3D end;
	};

	struct LinearBoundary{
		float start;
		float end;
	};

}

#endif
