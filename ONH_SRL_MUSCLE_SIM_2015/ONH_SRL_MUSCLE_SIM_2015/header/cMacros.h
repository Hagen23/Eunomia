#pragma once

//#define _USE_MATH_DEFINES
#include <qmath.h>
#include <qmatrix4x4.h>

#define DEG2RAD M_PI/180.0f

#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

#ifndef SHADER_MATERIAL
#define SHADER_MATERIAL
// This is for a shader uniform block
struct ShaderMaterial
{
	//[0]diffuse,[1]ambient,[2]specular,[3]emmisive
	float		diffuse[4];
	float		ambient[4];
	float		specular[4];
	float		emissive[4];
	QMatrix4x4	material;
	float		shininess;
	float		opacity;
	int			texCount;
};
#endif /*SHADER_MATERIAL*/

#ifndef AISGL_MINMAX
#define AISGL_MINMAX
#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)
#endif /*AISGL_MINMAX*/

#ifndef ARM_PARTS
#define ARM_PARTS
static enum ARM_PART{ AP_ANCONEUS, AP_BRACHIALIS, AP_BRACHIORADIALIS, AP_PRONATOR_TERES, AP_BICEPS_BRACHII, AP_TRICEPS_BRACHII, AP_OTHER, AP_BONES };
#endif /*ARM_PARTS*/

#ifndef NUM_ARM_PARTS
	#define NUM_ARM_PARTS 8
#endif /*NUM_ARM_PARTS*/

