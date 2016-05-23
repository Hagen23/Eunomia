#ifndef LB_UTIL
#define LB_UTIL

#include <cmath>

//struct int3
//{
//	int x, y, z;
//
//	inline int3& operator +=(const int3& rh)
//	{
//		x += rh.x;
//		y += rh.y;
//		z += rh.z;
//		return *this;
//	}
//
//	inline int3 operator+(const int3& rhs)
//	{
//		*this += rhs;
//		return *this;
//	}
//
//	inline int3& operator -=(const int3& rh)
//	{
//		x -= rh.x;
//		y -= rh.y;
//		z -= rh.z;
//		return *this;
//	}
//
//	inline int3 operator-(const int3& rhs)
//	{
//		*this -= rhs;
//		return *this;
//	}
//
//	inline int3 operator/(const int& rhs)
//	{
//		*this /= rhs;
//		return *this;
//	}
//
//	inline int3& operator /=(const int& rh)
//	{
//		x /= rh;
//		y /= rh;
//		z /= rh;
//		return *this;
//	}
//
//	bool operator ==(const int3& rh)
//	{
//		return (x == rh.x && y == rh.y && z == rh.z);
//	}
//
//	bool operator !=(const int3& rh)
//	{
//		return (x != rh.x && y != rh.y && z != rh.z);
//	}
//};

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

//struct float3
//{
//	float x, y, z;
//
//	inline float3& operator +=(const float3& rh)
//	{
//		x += rh.x;
//		y += rh.y;
//		z += rh.z;
//		return *this;
//	}
//
//	inline float3 operator+(const float3& rhs)
//	{
//		*this += rhs;
//		return *this;
//	}
//
//	inline float3& operator -=(const float3& rh)
//	{
//		x -= rh.x;
//		y -= rh.y;
//		z -= rh.z;
//		return *this;
//	}
//
//	inline float3 operator-(const float3& rhs)
//	{
//		*this -= rhs;
//		return *this;
//	}
//
//	inline float3 operator/(const float& rhs)
//	{
//		*this /= rhs;
//		return *this;
//	}
//
//	inline float3& operator /=(const float& rh)
//	{
//		x /= rh;
//		y /= rh;
//		z /= rh;
//		return *this;
//	}
//
//	bool operator ==(const float3& rh)
//	{
//		return (x == rh.x && y == rh.y && z == rh.z);
//	}
//
//	bool operator !=(const float3& rh)
//	{
//		return (x != rh.x && y != rh.y && z != rh.z);
//	}
//};

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

//static inline float dot(int3 a, float3 b)
//{
//	return a.x*b.x + a.y*b.y + a.z * b.z;
//}
//
//static inline float dot(float3 a, float3 b)
//{
//	return a.x*b.x + a.y*b.y + a.z * b.z;
//}
//
//static inline float3 float3_ScalarMultiply(const float s, const int3 v)
//{
//	return float3{ s*v.x, s*v.y, s*v.z, };
//}
//
//static inline float3 float3_ScalarMultiply(const float s, const float3 v)
//{
//	return float3{ s*v.x, s*v.y, s*v.z, };
//}
//
//static inline float float3_Norm(const float3 v)
//{
//	return sqrtf(dot(v, v));
//}

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

#endif