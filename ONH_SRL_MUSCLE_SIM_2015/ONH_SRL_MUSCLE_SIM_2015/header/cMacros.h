#pragma once

//#define _USE_MATH_DEFINES
#include <qmath.h>
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
	float	diffuse[4];
	float	ambient[4];
	float	specular[4];
	float	emissive[4];
	float	material[16];
	float	shininess;
	int		texCount;
};
#endif /*SHADER_MATERIAL*/

#ifndef AISGL_MINMAX
#define AISGL_MINMAX
#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)
#endif /*AISGL_MINMAX*/
