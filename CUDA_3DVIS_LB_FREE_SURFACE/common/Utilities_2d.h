#ifndef LB_UTIL
#define LB_UTIL

#include <cmath>

struct int3
{};

struct int2
{
	int x, y;

	inline int2& operator +=(const int2& rh)
	{
		x += rh.x;
		y += rh.y;
		return *this;
	}

	inline int2 operator+(const int2& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline int2& operator -=(const int2& rh)
	{
		x -= rh.x;
		y -= rh.y;
		return *this;
	}

	inline int2 operator-(const int2& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline int2 operator/(const int& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline int2& operator /=(const int& rh)
	{
		x /= rh;
		y /= rh;
		return *this;
	}

	bool operator ==(const int2& rh)
	{
		return (x == rh.x && y == rh.y);
	}

	bool operator !=(const int2& rh)
	{
		return (x != rh.x && y != rh.y);
	}
};

struct float2
{
	// The z is just for the display with opengl
	float x, y, z;

	inline float2& operator +=(const float2& rh)
	{
		x += rh.x;
		y += rh.y;
		return *this;
	}

	inline float2 operator+(const float2& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline float2& operator -=(const float2& rh)
	{
		x -= rh.x;
		y -= rh.y;
		return *this;
	}

	inline float2 operator-(const float2& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline float2 operator/(const float& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline float2& operator /=(const float& rh)
	{
		x /= rh;
		y /= rh;
		return *this;
	}

	bool operator ==(const float2& rh)
	{
		return (x == rh.x && y == rh.y);
	}

	bool operator !=(const float2& rh)
	{
		return (x != rh.x && y != rh.y);
	}
};

template<typename T> 
struct vector2d
{
	T x, y;

	inline T& operator +=(const T& rh)
	{
		x += rh.x;
		y += rh.y;
		return *this;
	}

	inline T operator+(const T& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline T& operator -=(const T& rh)
	{
		x -= rh.x;
		y -= rh.y;
		return *this;
	}

	inline T operator-(const T& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline T operator/(const int& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline T& operator /=(const int& rh)
	{
		x /= rh;
		y /= rh;
		return *this;
	}

	bool operator ==(const T& rh)
	{
		return (x == rh.x && y == rh.y);
	}

	bool operator !=(const T& rh)
	{
		return (x != rh.x && y != rh.y);
	}

	inline T dot(T a)
	{
		return x * a.x + y * a.y;
	}

	inline T dot(int2 a)
	{
		return a.x * x + a.y * y;
	}

	inline T ScalarMultiply(const double s)
	{
		return double3{ s* x, s* y };
	}

	inline T Norm()
	{
		return sqrtl(this->dot(this));
	}
};

static inline float dot(int2 a, float2 b)
{
	return a.x*b.x + a.y*b.y;
}

static inline float dot(float2 a, float2 b)
{
	return a.x*b.x + a.y*b.y;
}

static inline float2 float2_ScalarMultiply(const float s, const int2 v)
{
	return float2{ s*v.x, s*v.y };
}

static inline float2 float2_ScalarMultiply(const float s, const float2 v)
{
	return float2{ s*v.x, s*v.y };
}

static inline float float2_Norm(const float2 v)
{
	return sqrtf(dot(v, v));
}
//
//struct double3
//{
//	double x, y, z;
//
//	inline double3& operator +=(const double3& rh)
//	{
//		x += rh.x;
//		y += rh.y;
//		z += rh.z;
//		return *this;
//	}
//
//	inline double3 operator+(const double3& rhs)
//	{
//		*this += rhs;
//		return *this;
//	}
//
//	inline double3& operator -=(const double3& rh)
//	{
//		x -= rh.x;
//		y -= rh.y;
//		z -= rh.z;
//		return *this;
//	}
//
//	inline double3 operator-(const double3& rhs)
//	{
//		*this -= rhs;
//		return *this;
//	}
//
//	inline double3 operator/(const double& rhs)
//	{
//		*this /= rhs;
//		return *this;
//	}
//
//	inline double3& operator /=(const double& rh)
//	{
//		x /= rh;
//		y /= rh;
//		z /= rh;
//		return *this;
//	}
//
//	bool operator ==(const double3& rh)
//	{
//		return (x == rh.x && y == rh.y && z == rh.z);
//	}
//
//	bool operator !=(const double3& rh)
//	{
//		return (x != rh.x && y != rh.y && z != rh.z);
//	}
//};
//
struct double2
{
	// The z is just for the display with opengl
	double x, y, z;

	inline double2& operator +=(const double2& rh)
	{
		x += rh.x;
		y += rh.y;
		return *this;
	}

	inline double2 operator+(const double2& rhs)
	{
		*this += rhs;
		return *this;
	}

	inline double2& operator -=(const double2& rh)
	{
		x -= rh.x;
		y -= rh.y;
		return *this;
	}

	inline double2 operator-(const double2& rhs)
	{
		*this -= rhs;
		return *this;
	}

	inline double2 operator/(const double& rhs)
	{
		*this /= rhs;
		return *this;
	}

	inline double2& operator /=(const double& rh)
	{
		x /= rh;
		y /= rh;
		return *this;
	}

	bool operator ==(const double2& rh)
	{
		return (x == rh.x && y == rh.y);
	}

	bool operator !=(const double2& rh)
	{
		return (x != rh.x && y != rh.y);
	}
};

//static inline double dot(int3 a, double3 b)
//{
//	return a.x*b.x + a.y*b.y + a.z * b.z;
//}
//
//static inline double dot(double3 a, double3 b)
//{
//	return a.x*b.x + a.y*b.y + a.z * b.z;
//}
//
//static inline double3 double3_ScalarMultiply(const double s, const int3 v)
//{
//	return double3{ s*v.x, s*v.y, s*v.z, };
//}
//
//static inline double3 double3_ScalarMultiply(const double s, const double3 v)
//{
//	return double3{ s*v.x, s*v.y, s*v.z, };
//}
//
//static inline double double3_Norm(const double3 v)
//{
//	return sqrtl(dot(v, v));
//}

static inline double dot(int2 a, double2 b)
{
	return a.x*b.x + a.y*b.y;
}

static inline double dot(double2 a, double2 b)
{
	return a.x*b.x + a.y*b.y;
}

static inline double2 double2_ScalarMultiply(const double s, const int2 v)
{
	return double2{ s*v.x, s*v.y };
}

static inline double2 double2_ScalarMultiply(const double s, const double2 v)
{
	return double2{ s*v.x, s*v.y };
}

static inline double double2_Norm(const double2 v)
{
	return sqrtl(dot(v, v));
}

#endif